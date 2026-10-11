"""Pure helpers for the MMMU-Pro standard (10 options) vision study.

Nothing here reads the dataset, so local tests need no data. The manifest builder, and later
the runners, share these functions so that the rules registered in
docs/plan/vision_mmmu_pro_study.md have one implementation.
"""

from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Hashable, Mapping, Sequence

from authority_allocation.prompts import PROMPT_VARIABLE

# Prompts frozen for the MMMU-Pro study, dev and test alike (docs/plan/vision_mmmu_pro_study.md
# section 5). Hashes are over LF line endings, as the repository stores them. Changing a frozen
# prompt starts a new study on this dataset, not a new condition.
VL_SOLVE_PROMPT = "conditions/c033/prompts/vl_solve_v1.txt"
VL_REVIEW_PROMPT = "conditions/c035/prompts/vl_review_v1.txt"
FROZEN_PROMPTS = {
    VL_SOLVE_PROMPT: "7f9287ab503df3ec98e2fb4419065213b20f550239f42e4a0f2d029d8237408b",
    VL_REVIEW_PROMPT: "b8c0474d81222a3c889c4187d937bf41e05ba7a4ff59b98a7debb4b6b8bfffa3",
}

OPTION_LETTERS = "ABCDEFGHIJ"
MAX_IMAGES = 7
IMAGE_TAG = re.compile(r"<image\s+(\d+)>")

# Qwen3-VL: 16-pixel patches merged 2x2, so one visual token covers 32x32 pixels.
QWEN3_VL_FACTOR = 32
QWEN3_VL_DEFAULT_MIN_PIXELS = 65536
QWEN3_VL_DEFAULT_MAX_PIXELS = 16777216
MAX_ASPECT_RATIO = 200

# A registered answer line: "Answer:" at the start of a line, after optional markdown
# decoration and an optional "Final "; the letter must follow on the same line.
_ANSWER_LINE = re.compile(
    r"^[ \t>#*_\-]*(?:[Ff]inal[ \t]+)?Answer(?:\*\*)?[ \t]*:"
    r"[ \t*]*(?:\$?[ \t]*\\boxed\{|[\(\[\$])?[ \t]*([A-J])?(?![A-Za-z])",
    re.MULTILINE)


def require_frozen_prompt(root: Path, relative: str) -> None:
    if relative not in FROZEN_PROMPTS:
        raise SystemExit(f"{relative} is not a frozen MMMU-Pro prompt")
    data = (root / relative).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != FROZEN_PROMPTS[relative]:
        raise SystemExit(f"frozen prompt {relative} no longer has its registered hash")


def parse_options(raw: Any) -> list[str]:
    """The source stores options as a Python-literal string; a list is accepted as is."""
    value = ast.literal_eval(raw) if isinstance(raw, str) else raw
    if not isinstance(value, (list, tuple)) or not all(isinstance(o, str) for o in value):
        raise ValueError("options are not a list of strings")
    return list(value)


def render_options(options: Sequence[str]) -> str:
    """MMMU-Pro's official rendering: one "A. option" line per option."""
    if len(options) > len(OPTION_LETTERS):
        raise ValueError("more options than letters")
    return "\n".join(f"{letter}. {option}" for letter, option in zip(OPTION_LETTERS, options))


def referenced_images(text: str) -> list[int]:
    """Image numbers in order of first reference; an image referenced twice appears once."""
    order: list[int] = []
    for match in IMAGE_TAG.finditer(text):
        number = int(match.group(1))
        if number not in order:
            order.append(number)
    return order


def answer_is_option(answer: str, n_options: int) -> bool:
    return len(answer) == 1 and answer in OPTION_LETTERS[:n_options]


def smart_resize(height: int, width: int, *, min_pixels: int, max_pixels: int,
                 factor: int = QWEN3_VL_FACTOR) -> tuple[int, int]:
    """Qwen2-VL/Qwen3-VL image resizing (transformers qwen2_vl smart_resize), restated."""
    if min(height, width) <= 0:
        raise ValueError("empty image")
    if max(height, width) / min(height, width) > MAX_ASPECT_RATIO:
        raise ValueError("aspect ratio over 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def visual_tokens(height: int, width: int, *, min_pixels: int = QWEN3_VL_DEFAULT_MIN_PIXELS,
                  max_pixels: int = QWEN3_VL_DEFAULT_MAX_PIXELS,
                  factor: int = QWEN3_VL_FACTOR) -> int:
    """Visual tokens one image becomes after resizing and 2x2 merging."""
    h_bar, w_bar = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels,
                                factor=factor)
    return (h_bar // factor) * (w_bar // factor)


def allocate_largest_remainder(sizes: Mapping[Hashable, int], total: int) -> dict[Hashable, int]:
    """Split ``total`` across strata in proportion to their sizes, summing exactly to it.

    Each stratum gets the floor of its quota; the remaining units go to the largest
    fractional parts, ties broken by sorted stratum key.
    """
    population = sum(sizes.values())
    if not 0 <= total <= population:
        raise ValueError("total must lie between 0 and the population size")
    keys = sorted(sizes)
    quotas = {key: total * sizes[key] / population for key in keys}
    allocation = {key: math.floor(quotas[key]) for key in keys}
    remaining = total - sum(allocation.values())
    by_remainder = sorted(keys, key=lambda key: (-(quotas[key] - allocation[key]), key))
    for key in by_remainder[:remaining]:
        allocation[key] += 1
    return allocation


def parse_answer_letter(text: str | None, n_options: int = len(OPTION_LETTERS)) -> str | None:
    """The registered answer, read from the last answer line only.

    An answer line starts with "Answer:" (after optional markdown decoration such as "**",
    "#", "-" or ">", and an optional "Final "). The letter follows on the same line, after
    optional "**", "(", "[", "$" or "\\boxed{", and no other letter may follow it. Only the
    last answer line counts, and "Answer:" in mid-sentence never does, so a reviewer quoting
    the proposed solution's answer is not read as its own. A last answer line without a
    letter, or with a letter outside this question's options, is unparseable (None). There
    is no random fallback.
    """
    located = locate_answer_letter(text, n_options)
    return None if located is None else located[0]


def locate_answer_letter(text: str | None,
                         n_options: int = len(OPTION_LETTERS)) -> tuple[str, int] | None:
    """parse_answer_letter's letter together with its character offset in ``text``."""
    if not text:
        return None
    matches = list(_ANSWER_LINE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    letter = last.group(1)
    if letter is None or not answer_is_option(letter, n_options):
        return None
    return letter, last.start(1)


def official_parse_without_random(response: str | None, options: Sequence[str]) -> str | None:
    """MMMU-Pro's evaluate.parse_multi_choice_response, with its random fallback removed.

    Used only to compare with published numbers; the registered reading is
    parse_answer_letter. Where the official code picks a random option this returns None.
    """
    all_choices = list(OPTION_LETTERS[:len(options)])
    index2ans = dict(zip(all_choices, options))
    if not isinstance(response, str):
        return None
    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        answer_str = response[last_answer_pos + len("Answer:"):].strip()
        matching = [option for option in all_choices if option in answer_str]
        if len(matching) == 1:
            return matching[0]
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "
    index_ans, ans_with_brack, candidates = True, False, []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True
    if not candidates:
        candidates = [choice for choice in all_choices if f"{choice} " in response]
    if not candidates:
        candidates = [choice for choice in all_choices if f"{choice}." in response]
    if not candidates and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if index_ans:
        needle = (lambda c: f"({c})") if ans_with_brack else (lambda c: f" {c} ")
        starts = [response.rfind(needle(c)) for c in candidates]
    else:
        starts = [response.lower().rfind(index2ans[c].lower()) for c in candidates]
    return candidates[max(range(len(candidates)), key=lambda i: (starts[i], -i))]


def user_content_parts(template: str, *, question: str, options: Sequence[str],
                       y0: str | None = None) -> list[dict[str, Any]]:
    """The user turn as ordered text and image parts.

    Images enter only through the question and options: at the first reference to
    <image n>, the tag text is kept and the image follows it, so later references to the
    same tag stay resolvable; later references stay text only. The proposed solution y0 is
    inserted as plain text even if it contains image tags. Image parts carry the image
    number; the runner attaches the bytes.
    """
    variables = set(PROMPT_VARIABLE.findall(template))
    expected = {"question", "options"} | ({"y0"} if y0 is not None else set())
    if variables != expected:
        raise ValueError(f"template variables {sorted(variables)} != {sorted(expected)}")
    values = {"question": question, "options": render_options(options), "y0": y0}
    parts: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add_text(text: str) -> None:
        if not text:
            return
        if parts and parts[-1]["type"] == "text":
            parts[-1]["text"] += text
        else:
            parts.append({"type": "text", "text": text})

    position = 0
    for match in PROMPT_VARIABLE.finditer(template):
        add_text(template[position:match.start()])
        name, value = match.group(1), values[match.group(1)]
        if name == "y0":
            add_text(value)
        else:
            cursor = 0
            for tag in IMAGE_TAG.finditer(value):
                add_text(value[cursor:tag.end()])
                number = int(tag.group(1))
                if number not in seen:
                    seen.add(number)
                    parts.append({"type": "image", "number": number})
                cursor = tag.end()
            add_text(value[cursor:])
        position = match.end()
    add_text(template[position:])
    return parts


IMAGE_MIME = {"PNG": "image/png"}


def data_uri(data: bytes, image_format: str) -> str:
    """An image as sent to vLLM: its original bytes, base64 encoded, never re-encoded."""
    import base64

    mime = IMAGE_MIME.get(image_format)
    if mime is None:
        raise ValueError(f"unregistered image format {image_format!r}")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def openai_user_content(parts: Sequence[Mapping[str, Any]],
                        images: Mapping[int, tuple[bytes, str]]) -> list[dict[str, Any]]:
    """user_content_parts in OpenAI chat format; ``images`` maps number to (bytes, format)."""
    content: list[dict[str, Any]] = []
    for part in parts:
        if part["type"] == "text":
            content.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image":
            data, image_format = images[part["number"]]
            content.append({"type": "image_url", "image_url": {"url": data_uri(data, image_format)}})
        else:
            raise ValueError(f"unknown content part {part['type']!r}")
    return content


def image_numbers_in_order(parts: Sequence[Mapping[str, Any]]) -> list[int]:
    return [int(part["number"]) for part in parts if part["type"] == "image"]


def count_token(token_ids: Sequence[int] | None, token_id: int) -> int | None:
    return None if token_ids is None else sum(1 for value in token_ids if value == token_id)
