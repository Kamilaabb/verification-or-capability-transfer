# code_solve_v1

Protocol for the BigCodeBench code study (`docs/plan/code_bigcodebench_study.md`): each model
alone on the 832 development rows of `bigcodebench-b74c0d0-instruct-s20260917`, with
same-forward logit capture and scalar evidence written before grading. Reviews (GV) are
registered later with their own prompt and schemas.

- **Records.** `code_record_v1.json`: one solve call, derived from `text_mc_v1`'s
  `mc_record_v1` with `problem_metadata` (`entry_point`, `libs`) and no answer letter. Native
  traces use `protocols/router_v1/schemas/native_logit_trace_record_v1.json`.
- **Input.** The frozen solve prompt `conditions/c100/prompts/code_solve_v1.txt`, rendered by
  `authority_allocation.bigcodebench_study.user_message`; `{{problem}}` is the dataset's
  `instruct_prompt` verbatim.
- **Evidence.** `features.yaml` (`code_solve_scalar_v1`); `code_evidence_record_v1.json`.
- **Grading.** After evidence, on the CPU partition: `scripts/bigcodebench_grade.py` in the
  pinned evaluate container with networking disabled
  (`protocols/authority_v6/grader_bigcodebench_v1.yaml`). The container extracts the program
  from the complete output (`bigcodebench.sanitize`, must define the entry point); no program
  is `model_invalid`, u=0; a timeout is u=0; any `grader_error` rejects the run.
- **Never shown to any model or to evidence:** the task's tests, canonical solution, grades.
