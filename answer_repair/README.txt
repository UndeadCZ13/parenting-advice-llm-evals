ParentBench Answer Repair Workspace
==================================

This folder contains a standalone workflow for repairing truncated or otherwise
damaged answer outputs without modifying the main `scripts/` pipeline.

Workflow
--------
1. Edit `config/truncation_rules.toml` if you want to change the candidate
   detection rules.
2. Run `scripts/01_scan_candidates.py` to generate a manual review list.
3. Mark the review list by filling `manual_action`:
   - `1` / `retry` / `rerun` / `repair` = export this item for rerun
   - leave blank = ignore for now
   - optional notes can go into `manual_notes`
   The review file may be kept as `.csv` or `.numbers`.
4. Run `scripts/02_export_generation_prompts.py` to export the exact
   generation prompts used by the normal pipeline.
5. Choose one of two generation paths:
   - Manual copy/paste path:
     use `workspace/prompt_packets/` or `workspace/prompt_txt/`, then save the
     recovered answers under `workspace/manual_replies/txt/`
   - Optional scripted batch path:
     run `scripts/05_batch_generate_replies.py`, which writes grouped reply
     JSONL files under `workspace/manual_replies/jsonl/`
6. Run `scripts/03_validate_manual_replies.py` to normalize and validate the
   replies.
7. Run `scripts/04_apply_manual_repairs.py` to patch the original
   `data/model_outputs/*.jsonl` files in place. Only selected rows are replaced.
8. Run `scripts/06_rerun_judges.py` to incrementally rerun rubric and style
   judge outputs for the repaired answer rows only.

Key files
---------
- `config/truncation_rules.toml`
  Machine-readable truncation rules. The scan and validation scripts both read
  this file.

- `TRUNCATION_RULES.txt`
  Short explanation of what each rule means and how to adjust it.

- `workspace/review_lists/`
  Manual review CSVs produced by the scan step.

- `workspace/prompt_jsonl/` and `workspace/prompt_txt/`
  Exact generation prompts exported from the selected review rows.

- `workspace/prompt_packets/`
  One combined TXT packet per source file, designed for manual copy/paste.

- `workspace/manual_replies/`
  Place manually recovered model answers here.

- `workspace/normalized_replies/`
  Cleaned and validated replies produced by step 3.

- `workspace/manifests/`
  Audit manifests and per-run logs.

- `workspace/judge_rerun/`
  Incremental judge rerun outputs and manifests created by step 8.

Manual reply formats
--------------------
Preferred JSONL format:

  workspace/manual_replies/jsonl/en_ollama_glm46.jsonl

Each line:

  {"scenario_uid": "pb_v0_0026", "answer": "..."}

Optional fields:

  {"scenario_uid": "pb_v0_0026", "answer_raw": "...", "answer": "..."}

TXT format for copy-paste workflow:

  workspace/manual_replies/txt/en_ollama_glm46/pb_v0_0026.txt

The TXT file should contain the final answer text only.

Important notes
---------------
- Step 7 repairs answer outputs. Step 8 reruns only the affected rubric/style
  judge rows and merges them back into the existing judge files.
- The repair apply step writes back to the original answer files using an
  atomic replace, but it also records manifests and backups of the replaced rows.
- The scan step is deliberately candidate-oriented. It is designed to produce a
  review list, not to make a final automatic decision.
