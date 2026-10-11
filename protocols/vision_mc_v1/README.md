# vision_mc_v1

Protocol for the MMMU-Pro vision study (`docs/plan/vision_mmmu_pro_study.md`): G alone, V
alone and GV on multimodal multiple-choice questions, with same-forward logit capture and
scalar evidence written before grading.

- **Records.** `vl_record_v1.json` for one solve call (c033, c034), `vl_review_record_v1.json`
  for V reviewing G (c035). Both are `g_record_single_v3` plus `visual_input`: the images in
  the order the model saw them, each byte-identical to the dataset's image audit, and the
  `<|image_pad|>` count of the served prompt, which must equal what the registered pixel
  limits imply. Native traces use `protocols/router_v1/schemas/native_logit_trace_record_v1.json`.
- **Input.** `authority_allocation.mmmu_pro.user_content_parts`: an image follows the first
  `<image n>` tag in the question or options; the tag text stays; a proposed solution is
  plain text. Images are sent as their original PNG bytes.
- **Answer.** `parse_answer_letter`: the letter on the last line starting with `Answer:`,
  within the row's options, no random fallback. Grading is letter equality with the
  reference answer, read only after evidence is written.
- **Delivery (c035).** Primary: V's letter, G's letter when V's is unparseable. Strict:
  V unparseable counts as wrong. Both are reported with the number of fallback rows.
- **Evidence.** `features.yaml` (`vision_mc_scalar_v1`); `vl_evidence_record_v1.json`.
- **Never shown to any model or to evidence:** `answer`, `explanation`, grades.
