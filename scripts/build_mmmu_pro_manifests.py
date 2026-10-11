"""Build the frozen dev and sealed test manifests for MMMU-Pro standard (10 options).

Written to run once, before any model has seen any of these questions. The design is
registered in docs/plan/vision_mmmu_pro_study.md section 3.

Source is MMMU/MMMU_Pro at a pinned revision, config "standard (10 options)", whose only
split holds 1,730 rows. Exclusions are decided from the source data alone, never from any
model output:

    duplicate_id                 an id that occurs more than once; every copy is dropped
    duplicate_question           identical question text and options on several rows; every
                                 copy is dropped, so one question never sits on both sides
    options_unparseable          options are not a list of strings
    option_count_out_of_range    fewer than 2 or more than 10 options
    answer_not_an_option         the reference answer is not the letter of an existing option
    no_referenced_image          neither question nor options reference any <image n>
    referenced_image_missing     a referenced image_n is empty, or n exceeds 7
    image_unreadable             a referenced image has no readable size
    image_aspect_ratio_over_200  Qwen image preprocessing would refuse the image

Split: dev takes 1,000 rows, test everything else. Strata are (subject, id prefix), the
prefix recording which split of the original MMMU a question came from. Dev quotas are
proportional to stratum size by largest remainder. Rows are sorted by id and each stratum,
in sorted key order, is permuted once by one PCG64 stream; its first quota rows go to dev.
Dev and test are then each permuted once more for run order only.

The same pass writes an image audit that reads only image sizes and bytes, never question
text beyond its <image n> tags, answers or explanations: the sha256 of every referenced
image, and how many visual tokens each image and row would become under several caps, so
the registered pixel cap can be chosen from data (plan section 4).

Manifests are id-only; content is joined from the pinned revision at run time.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from datasets import Image as ImageFeature, load_dataset
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from authority_allocation import mmmu_pro  # noqa: E402

REPO = "MMMU/MMMU_Pro"
REVISION = "563f3e84bb3b90893083a1f039cfa13077f2302b"
CONFIG = "standard (10 options)"
EXPECTED_ROWS = 1730
DEV_ROWS = 1000
SPLIT_SEED = 20260926
DEV_ORDER_SEED = 20260927
TEST_ORDER_SEED = 20260928
ORIGIN_PREFIXES = ("dev", "validation", "test")
# Caps on visual tokens per image audited before one is registered; None is the default.
TOKEN_CAPS = (768, 1280, 2048, 4096, None)


def origin_of(problem_id: str) -> str:
    prefix = problem_id.split("_", 1)[0]
    if prefix not in ORIGIN_PREFIXES:
        raise RuntimeError(f"unexpected id prefix: {problem_id}")
    return prefix


def question_key(question: str, options: list[str]) -> str:
    payload = json.dumps([question.strip(), [o.strip() for o in options]], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def max_pixels_for(cap: int | None) -> int:
    if cap is None:
        return mmmu_pro.QWEN3_VL_DEFAULT_MAX_PIXELS
    return cap * mmmu_pro.QWEN3_VL_FACTOR ** 2


def cap_name(cap: int | None) -> str:
    return "default" if cap is None else str(cap)


def quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    points = {"min": 0, "p50": 50, "p90": 90, "p99": 99, "max": 100}
    return {name: float(np.percentile(array, q)) for name, q in points.items()}


def transformers_smart_resize():
    """The library's own smart_resize, to cross-check the restated one, when importable."""
    try:
        import transformers
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        smart_resize(448, 640, factor=mmmu_pro.QWEN3_VL_FACTOR,
                     min_pixels=mmmu_pro.QWEN3_VL_DEFAULT_MIN_PIXELS,
                     max_pixels=mmmu_pro.QWEN3_VL_DEFAULT_MAX_PIXELS)
    except Exception:
        return None, None
    return smart_resize, transformers.__version__


def write_manifest(path: Path, rows: list[dict]) -> dict:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    origins = collections.Counter(row["origin"] for row in rows)
    print(f"{path.name:<12} rows={len(rows):<5} origins={dict(sorted(origins.items()))} sha256={digest}")
    return {"rows": len(rows), "sha256": digest,
            "counts_by_origin": dict(sorted(origins.items())),
            "counts_by_subject": dict(sorted(collections.Counter(r["subject"] for r in rows).items()))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} is not empty; manifests are written once")

    source = load_dataset(REPO, CONFIG, revision=REVISION, split="test")
    if len(source) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} rows, found {len(source)}")
    image_columns = [f"image_{n}" for n in range(1, mmmu_pro.MAX_IMAGES + 1)]
    for column in image_columns:
        source = source.cast_column(column, ImageFeature(decode=False))
    library_resize, transformers_version = transformers_smart_resize()

    id_count = collections.Counter(source["id"])
    parsed_options: dict[int, list[str] | None] = {}
    key_count: collections.Counter[str] = collections.Counter()
    for index, row in enumerate(source):
        try:
            options = mmmu_pro.parse_options(row["options"])
        except (ValueError, SyntaxError):
            options = None
        parsed_options[index] = options
        if options is not None:
            key_count[question_key(row["question"], options)] += 1

    excluded: dict[str, list[str]] = collections.defaultdict(list)
    kept: list[dict] = []
    image_audit: list[dict] = []
    unreferenced_images = 0
    for index, row in enumerate(source):
        pid = str(row["id"])
        origin = origin_of(pid)
        options = parsed_options[index]
        if id_count[pid] > 1:
            excluded["duplicate_id"].append(pid)
            continue
        if options is None:
            excluded["options_unparseable"].append(pid)
            continue
        if key_count[question_key(row["question"], options)] > 1:
            excluded["duplicate_question"].append(pid)
            continue
        if not 2 <= len(options) <= len(mmmu_pro.OPTION_LETTERS):
            excluded["option_count_out_of_range"].append(pid)
            continue
        if not mmmu_pro.answer_is_option(str(row["answer"]).strip(), len(options)):
            excluded["answer_not_an_option"].append(pid)
            continue
        numbers = mmmu_pro.referenced_images(row["question"] + "\n" + mmmu_pro.render_options(options))
        if not numbers:
            excluded["no_referenced_image"].append(pid)
            continue
        if any(n > mmmu_pro.MAX_IMAGES or not row[f"image_{n}"] for n in numbers):
            excluded["referenced_image_missing"].append(pid)
            continue
        images, reason = [], None
        for n in numbers:
            data = row[f"image_{n}"].get("bytes")
            if not data:
                reason = "image_unreadable"
                break
            try:
                with Image.open(io.BytesIO(data)) as image:
                    width, height = image.size
                    mode, fmt = image.mode, image.format
            except Exception:
                reason = "image_unreadable"
                break
            if min(width, height) <= 0:
                reason = "image_unreadable"
                break
            if max(width, height) / min(width, height) > mmmu_pro.MAX_ASPECT_RATIO:
                reason = "image_aspect_ratio_over_200"
                break
            tokens = {}
            for cap in TOKEN_CAPS:
                max_pixels = max_pixels_for(cap)
                tokens[cap_name(cap)] = mmmu_pro.visual_tokens(height, width, max_pixels=max_pixels)
                if library_resize is not None:
                    ours = mmmu_pro.smart_resize(height, width, max_pixels=max_pixels,
                                                 min_pixels=mmmu_pro.QWEN3_VL_DEFAULT_MIN_PIXELS)
                    theirs = library_resize(height, width, factor=mmmu_pro.QWEN3_VL_FACTOR,
                                            min_pixels=mmmu_pro.QWEN3_VL_DEFAULT_MIN_PIXELS,
                                            max_pixels=max_pixels)
                    if tuple(ours) != tuple(theirs):
                        raise RuntimeError(f"smart_resize disagrees with transformers on {pid} image_{n}")
            images.append({"problem_id": pid, "image_number": n, "sha256": hashlib.sha256(data).hexdigest(),
                           "width": width, "height": height, "pixels": width * height,
                           "mode": mode, "format": fmt, "visual_tokens": tokens})
        if reason is not None:
            excluded[reason].append(pid)
            continue
        unreferenced_images += sum(1 for column in image_columns
                                   if row[column] and int(column.split("_")[1]) not in numbers)
        image_audit.extend(images)
        kept.append({"problem_id": pid, "subject": str(row["subject"]), "origin": origin,
                     "n_images": len(numbers), "n_options": len(options)})

    kept.sort(key=lambda r: r["problem_id"])
    strata: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in kept:
        strata[(row["subject"], row["origin"])].append(row)
    quotas = mmmu_pro.allocate_largest_remainder({k: len(v) for k, v in strata.items()}, DEV_ROWS)
    rng = np.random.Generator(np.random.PCG64(SPLIT_SEED))
    dev, test = [], []
    for key in sorted(strata):
        rows = strata[key]
        order = rng.permutation(len(rows))
        chosen = [rows[i] for i in order]
        dev.extend(chosen[:quotas[key]])
        test.extend(chosen[quotas[key]:])
    if {r["problem_id"] for r in dev} & {r["problem_id"] for r in test} or len(dev) != DEV_ROWS:
        raise RuntimeError("split is inconsistent")
    ordered = {}
    for name, rows, seed in (("dev", dev, DEV_ORDER_SEED), ("test", test, TEST_ORDER_SEED)):
        rows = sorted(rows, key=lambda r: r["problem_id"])
        order = np.random.Generator(np.random.PCG64(seed)).permutation(len(rows))
        ordered[name] = [rows[i] for i in order]

    row_tokens = collections.defaultdict(lambda: collections.defaultdict(int))
    for image in image_audit:
        for cap, tokens in image["visual_tokens"].items():
            row_tokens[cap][image["problem_id"]] += tokens
    audit_summary = {
        "images": len(image_audit),
        "unreferenced_images_not_input": unreferenced_images,
        "images_per_row": dict(sorted(collections.Counter(r["n_images"] for r in kept).items())),
        "image_pixels": quantiles([i["pixels"] for i in image_audit]),
        "caps": {},
    }
    for cap in TOKEN_CAPS:
        name = cap_name(cap)
        resized = sum(1 for i in image_audit
                      if i["visual_tokens"][name] < i["visual_tokens"][cap_name(None)])
        audit_summary["caps"][name] = {
            "max_pixels": max_pixels_for(cap),
            "images_reduced_below_default": resized,
            "per_image_tokens": quantiles([i["visual_tokens"][name] for i in image_audit]),
            "per_row_tokens": quantiles(list(row_tokens[name].values())),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source": {"repo": REPO, "revision": REVISION, "config": CONFIG, "rows": len(source)},
        "numpy_version": np.__version__,
        "seeds": {"split": SPLIT_SEED, "dev_order": DEV_ORDER_SEED, "test_order": TEST_ORDER_SEED},
        "smart_resize_cross_checked_with_transformers": transformers_version,
        "excluded": {k: sorted(v) for k, v in excluded.items()},
        "excluded_counts": {k: len(v) for k, v in excluded.items()},
        "rows_kept": len(kept),
        "options_per_row": dict(sorted(collections.Counter(r["n_options"] for r in kept).items())),
        "dev_quota_by_stratum": {f"{s}|{o}": q for (s, o), q in sorted(quotas.items())},
        "image_audit": audit_summary,
        "manifests": {},
    }
    report["manifests"]["dev"] = write_manifest(args.output_dir / "dev.jsonl", ordered["dev"])
    report["manifests"]["test"] = write_manifest(args.output_dir / "test.jsonl", ordered["test"])
    (args.output_dir / "image_audit.jsonl").write_text(
        "".join(json.dumps(i, sort_keys=True) + "\n" for i in image_audit), encoding="utf-8")
    (args.output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("excluded:", report["excluded_counts"])
    print(json.dumps(audit_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
