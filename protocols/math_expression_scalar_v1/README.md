# Math Expression Scalar v1

This protocol is a pass-1-only development diagnostic. It repeats c014's frozen
Qwen3-4B generation setup and same-forward raw-logit capture, then writes scalar
evidence before grading. It does not run self-check pass 2 or train a router.

`sample_index=0` is unconditionally `y0`. Raw G output, sampled-token logprobs,
same-forward compact raw-logit summaries, math spans, scalar evidence and separate
grades are retained.

Math-related tokens are all generated content tokens whose UTF-8 byte ranges overlap
the union of spans returned by `math_expression_spans_v1`: explicit `$...$`,
`$$...$$`, `\(...\)`, `\[...\]`, registered math environments, balanced standalone
math commands, and conservative plain equations. Markdown code spans and fences are
excluded. Delimiters are excluded from character spans, but a tokenizer token that
merges a delimiter with expression content is included by the overlap rule.

For each selected token, `max_raw_logit` is the maximum pre-softmax vocabulary logit
from the exact vLLM forward that generated it. Log probability is the sampled token's
natural log probability. Empty math-token sets yield null math aggregates; they are
never imputed. `answer_parseable` is independently determined by the frozen
math-verify answer parser over the complete G output.

Evidence is written before grades. The scalar evidence file contains no gold answer,
grader output or correctness label. This development run supports descriptive feature
measurement only.

