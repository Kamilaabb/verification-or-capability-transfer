# Review Rarely Beats the Stronger Model: code

Anonymous code release for the submission *Review Rarely Beats the Stronger Model: Separating Verification
from Capability Transfer in LLM Pipelines*. It contains the code that produced every number and figure in the
paper. Per-question results are in `results/`: whether each model alone, each review direction, the
selector and JEV got each question right, with the token counts behind the cost analysis. Every statistic
in the paper can be recomputed from them on a CPU (see below). Raw model outputs are not included; they
will be released upon acceptance.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests      # 433 tests, no data or GPU needed
```

Generation additionally needs vLLM and the model snapshots named in each `conditions/<id>/condition.yaml`.
Code grading uses the BigCodeBench evaluation container (`slurm/bigcodebench_image.sbatch`).
Plotting needs matplotlib.

## Layout

| Path | Contents |
|---|---|
| `src/authority_allocation/` | prompts, answer parsing and grading, statistics (`claim_stats.py`, `resampling.py`), outcome levels (`outcome_levels.py`), cost model (`cost_model.py`) |
| `conditions/` | one frozen configuration per run: models, revisions, prompts, decoding |
| `datasets/`, `protocols/`, `replicates/` | dataset identities, record schemas, seeds |
| `pairs/` | the model pairs of each domain |
| `claims/` | the analysis cards H01–H11 (see `claims/README.md`) |
| `scripts/` | entry points, listed below |
| `slurm/` | the job scripts used on our cluster; paths are placeholders (`/path/to/project`, `/path/to/scratch`) |
| `tests/` | unit tests |
| `results/` | per-question results: `results_db/` (written by `build_results_db.py`) and `cost/` (written by `build_cost_rows.py`); large files are gzipped |

The repository also contains code from earlier pilot experiments (conditions and scripts named `c0xx`).
They are kept because later runners import shared code from them; the paper does not use their results.

## Recomputing the statistics from `results/` (CPU only)

```bash
gunzip -k results/results_db/db/*.gz results/cost/db/*.gz
python scripts/rerun_claims_audited.py --db results/results_db --cost results/cost --out out/claims
```

This runs step 3 below without the archives of model outputs and writes H01–H11 and the RQ2 summaries
to `out/claims`; the figure scripts of step 4 read that directory. Problem ids are those of the public
datasets (MATH, MMLU-Pro, BigCodeBench, ReClor, FOLIO, GPQA); no question text is included.

## Pipeline

Every model decodes greedily (thinking off for Qwen3, at most 4,096 output tokens).

**1. Each model alone, and review in both directions** (GPU)

| Domain | Data | Runner | Job script |
|---|---|---|---|
| Math | MATH levels 3–5 | `run_mathfull_solo.py`, `run_mathfull_review.py` | `mathfull_solo.sbatch`, `mathfull_review.sbatch` |
| Knowledge | MMLU-Pro | `run_mmlu_pro.py` | `mmlu_pro.sbatch` |
| Code | BigCodeBench-Instruct | `run_bigcodebench_solo.py`, `run_bigcodebench_review.py`, `bigcodebench_grade.py` | `bigcodebench_*.sbatch` |
| Reasoning | ReClor, FOLIO v2, GPQA Diamond | `run_nl_reasoning.py` | `nl_reasoning.sbatch` |

Each runner takes `--condition-id` and a stage (`check-runnable`, `smoke`, generation/`review`,
`reconcile`, `evidence`, `grade`, `analyze`). Question lists are built by `build_*_manifests.py`.
Every model output is written before it is graded.

**2. The two equal-call baselines**

- Selector, which picks one of the two solo answers: `gv_table_selector_fit.py` (math, knowledge),
  `code_selector_backfill.py` (code), `nl_selector_crossfit.py` (reasoning).
- JEV, a judge that sees both solo answers: `jev_domain_sweep.py` (math, knowledge), `code_jev_dev832.py`,
  `nl_jev_eval901.py`.

**3. Results database, statistics and cost** (CPU)

```bash
python scripts/build_results_db.py --out-dir $OUT/results_db      # per-question table from the archives
python scripts/build_cost_rows.py --db $OUT/results_db --artifacts artifacts --out-dir $OUT/cost
python scripts/rerun_claims_audited.py --db $OUT/results_db --cost $OUT/cost --out $OUT/claims
```

`rerun_claims_audited.py` runs all eleven cards: `evaluate_claim.py` (H01–H07),
`explain_conditions.py` (H08), `explain_outcome_levels.py` (H09) and `evaluate_cost_claims.py` (H10–H11).
It also runs `rq2_cells.py` and `rq2_cells_bootstrap.py`. Intervals come from a paired question
bootstrap with fixed directions (2,000 replicates). The pair-resampling sensitivity analysis
in the appendix comes from the same scripts.

**Further checks (appendix).**

| Check | Script |
|---|---|
| All eight cells (G alone, V alone, final answer) | `eight_cells_table.py` (reads only `results/`, so it runs on a CPU from this package) |
| Answers equal to G's, truncation, a selector without labels, the gap regression on held-out questions only | `review_followup_checks.py` |
| Self-review (conditions c386–c389) | `self_review_sanity.py`, `self_review_test_subset.py` |
| Commit-first review prompt (conditions c390–c393) | `prompt_variant_test_subset.py`, `prompt_variant_committed_vs_g.py` |
| Thinking on (Qwen3-8B) | `thinking_check_test500.py` (generation and analysis; `slurm/thinking_check_test500.sbatch`), `thinking_check_loss_breakdown.py` |
| When the reviewer keeps its own right answer | `v_abandons_right_answer.py` |

Except for `eight_cells_table.py`, they read the archived model outputs. Part of the thinking-on calls went
to a hosted Qwen3-8B; `thinking_check_test500.py` then reads the key from a file passed with `--api-key-file`
(no key is included).

**4. Figures**

| Figure file | Script |
|---|---|
| `fig1_overview` | `plot_overview.py` |
| `fig2_example` | `plot_example.py` (example chosen by `select_example_candidates.py`) |
| `fig4_deltas_vs_gap`, `fig3_outcome_levels` (appendix) | `plot_rq1.py` |
| `fig5_mechanism` | `plot_rq2.py` |
| `fig7_cost_accuracy` | `plot_rq3.py` |
| `fig6_convergence` (appendix) | `plot_convergence.py` |

Each script reads the `H*_result.json` and `rq2_cells*.json` files written in step 3;
pass their directory with `--claims` (or `--h`/`--cells`).

## Definitions used in the code

- Δ_G = accuracy after review − the generator's own accuracy.
- Δ_S = accuracy after review − the stronger model's own accuracy. The code and the result files call it
  Δ_C (`delta_C`, `dC`), its name in an earlier draft.
- gap = the stronger model's accuracy − the weaker model's accuracy.
- Cost per question = parameters (billions) × (input + output tokens).
