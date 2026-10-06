# LLM_TRY v10.12.5 — Nagato Candidate Preservation / Promotion Gate

This release adds a dedicated promotion path for the moderate data-nagato.txt candidate.

## Candidate gates

The candidate is accepted only when all of the following pass:

1. Internalized checkpoint binding preservation
2. v10.12.4 Nagato adaptation metadata check
3. Direct model-weight validation of all protected internalized concepts
4. Composite Teacher Fidelity for every protected concept
5. Persona retention: "あなたは誰ですか" -> "長門有希。"

## Evaluation only

Run:

python .\promote_nagato_candidate_v10125.py

Expected final status:

[nagato candidate gate: ..., status=PASS]
[candidate accepted: evaluation only; production unchanged]

## Promotion

Only after the evaluation passes:

python .\promote_nagato_candidate_v10125.py --promote

Promotion uses a backup/restore transaction and replaces the production checkpoint only after every gate passes.

## Audit

Results are appended to:

data/nagato_candidate_gate_v10125.jsonl

## Current repository note

The GitHub branch-creation action was unavailable while this change was prepared, so the v10.12.5 files were staged on v10.12.4. Create/switch the v10.12.5 branch locally before pulling these changes into the release branch if desired.
