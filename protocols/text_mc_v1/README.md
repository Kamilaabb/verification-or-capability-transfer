# text_mc_v1

Protocol for the MMLU-Pro knowledge study (`docs/plan/mmlu_pro_study.md`): G alone, V alone and
GV in both directions on text multiple-choice questions, with same-forward logit capture and
scalar evidence written before grading. It is `vision_mc_v1` without images.

- **Records.** `mc_record_v1.json` for one solve call (c070, c071), `mc_review_record_v1.json`
  for one model reviewing the other (c072, c073): `g_record_single_v3` with the answer letter
  and `problem_metadata` (category, origin, n_options). Native traces use
  `protocols/router_v1/schemas/native_logit_trace_record_v1.json`.
- **Input.** The frozen prompts, rendered by `authority_allocation.mmlu_pro.user_message`; the
  proposed solution is plain text.
- **Answer.** `parse_answer_letter`, as in the vision study; grading is letter equality, read
  only after evidence is written.
- **Delivery (c072, c073).** Primary: the reviewer's letter, the reviewed model's letter when
  the reviewer's is unparseable. Strict: reviewer unparseable counts as wrong. Both are
  reported with the number of fallback rows.
- **Evidence.** `features.yaml` (`text_mc_scalar_v1`); `mc_evidence_record_v1.json`.
- **Never shown to any model or to evidence:** `answer`, `answer_index`, `cot_content`, grades.
