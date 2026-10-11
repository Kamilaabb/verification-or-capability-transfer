# Math verifier scalar protocol v1

This development-only protocol reuses the frozen c017 generator records and native
same-forward traces, then runs one neutral verifier call per row. It does not grade,
rewrite, train a Controller, or read gold answers.

The verifier must independently classify the generator output as `CORRECT`,
`INCORRECT`, or `UNCERTAIN`. The prompt must not presume that the generator is wrong.
It may request only `KEEP` or `CRITIQUE_REFINE`.

Every parsed report is classified the same way whatever its verdict: whether the
verifier quoted a location, and if so whether the quotation occurs in the generator
output exactly once, occurs more than once, occurs once only after all whitespace is
removed from both texts, or does not occur. The verdict is a separate feature, so any
interaction between verdict and location is left to the Controller. Token-level
location evidence is taken only from a unique exact quotation; descriptions,
model-supplied offsets, approximate matches and forced guesses yield none. The
matched character and token spans are retained for audit.

The report parser requires the section headers in order and tolerates whitespace-only
deviations. A field's END marker may be omitted when the next section header, or the
end of output for the final answer, already delimits that field; field text may not
contain its own END marker or run into the next header. On c018 the verifier omitted
`END CRITIQUE` in 110 of 300 reports, disproportionately when it disagreed with the
generator, so requiring the marker would have discarded most objections; see
`docs/analysis/c018_report_format.md`. The parser never repairs or rewrites model
output; field spans index the raw text.

All sampled-token log probabilities are natural log probabilities from the generated
token trace. Every `mean_max_logit` is the arithmetic mean of the full-vocabulary raw
pre-softmax maximum at each selected token position. It is not a logarithm. Only the
actually emitted verdict has token evidence; the two unselected class-specific slots
are null rather than counterfactual scores.

Evidence is written without gold or outcome labels. The complete V text, token trace,
native raw-logit summaries, parsed field spans, math spans, and G/V localization spans
are retained outside the scalar payload.
