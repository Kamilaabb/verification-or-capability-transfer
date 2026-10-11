# Analysis cards

Each card fixes one question, its unit, its statistic and its decision rule before the report is
generated. The result sections are written by scripts, never by hand. Some descriptive fields
(`statement_zh`, comments) and the generated report text are in Chinese; field names, keys and
numbers are in English.

| Card | Question | Unit | Script |
|---|---|---|---|
| H01 | With the same pair, does the stronger model generating and the weaker reviewing beat the reverse? | pair | `evaluate_claim.py` |
| H02 | Does review raise the generator's accuracy (Δ_G > 0)? | direction | `evaluate_claim.py` |
| H03 | Does review fall short of the stronger model alone (Δ_S < 0)? | direction | `evaluate_claim.py` |
| H04 | On questions both models get right or both get wrong, does review gain more than it loses? | direction | `evaluate_claim.py` |
| H05 | When exactly one model is right, does the reviewed answer follow the generator? | direction | `evaluate_claim.py` |
| H06 | Does review make the two models' answers converge? | direction | `evaluate_claim.py` |
| H07 | Do the two models' independent answers leave room above the stronger model? | pair | `evaluate_claim.py` |
| H08 | How do Δ_G and Δ_S change with the capability gap and with headroom? | direction, pair | `explain_conditions.py` |
| H09 | Which of the five outcome levels does each direction reach? | direction | `explain_outcome_levels.py` |
| H10 | Accuracy and cost of each way of using a second model, against the stronger model alone | pair | `evaluate_cost_claims.py` |
| H11 | With the same two calls: selector against review in both directions | pair | `evaluate_cost_claims.py` |

`labels.yaml` maps datasets to domains and splits. `system_sources.json` pins the selector and JEV
archives by SHA-256.
