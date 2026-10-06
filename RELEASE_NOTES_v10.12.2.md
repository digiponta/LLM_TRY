# LLM_TRY v10.12.2 — Batch Repair Rollback / Candidate Checkpoint

v10.12.2 turns the v10.12.1 Preservation Gate into a safe checkpoint promotion workflow.

## Problem

Before v10.12.2, /trainbatch wrote directly to the production online checkpoint. If preservation later failed, the regressed checkpoint had already replaced the previous production model.

## Candidate workflow

/trainbatch now uses:

production checkpoint/state
  -> candidate checkpoint/state
  -> batch training
  -> Preservation Gate
     -> PASS: promote candidate checkpoint + candidate state
     -> FAIL: discard candidate checkpoint + candidate state

The currently loaded production model remains active until candidate validation succeeds.

## Candidate files

For the default online artifacts:

- model/model-gpu-v1.6.2-online.candidate.pt
- data/chat_learning_state.candidate.json

These are temporary artifacts.

## PASS path

After Preservation Gate PASS:

1. checkpoint/state promotion backups are created
2. candidate checkpoint replaces production checkpoint
3. candidate state replaces production state
4. if either promotion step fails, both production files are restored
5. promotion backups are removed
6. promoted checkpoint is loaded
7. checkpoint binding is refreshed
8. repair targets remain active until post-training re-query verifies them

Expected runtime:

[batch candidate promoted: model\model-gpu-v1.6.2-online.pt]
[batch state promoted: data\chat_learning_state.json]
[verification batch trained: 2 task(s); preservation=PASS; candidate=PROMOTED; ...]

## FAIL path

After Preservation Gate FAIL:

- candidate checkpoint is deleted
- candidate state is deleted
- production checkpoint is retained
- production learning state remains in retrain/pending form
- verification tasks remain retrain
- runtime continues on the previous production model

Expected runtime:

[batch rollback: preservation=FAIL; candidate checkpoint/state discarded]
[production checkpoint retained: ...]
[verification tasks remain retrain; adjust training/replay before retry]

## Transaction safety

Checkpoint and learning-state promotion are treated as one logical transaction. Pre-promotion backups are restored if partial promotion fails.

## Regression

Run:

python .\run_batch_repair_candidate_rollback_v10122.py

Expected:

STATUS : BATCH_REPAIR_CANDIDATE_ROLLBACK_PASS

Also rerun:

python .\run_batch_repair_preservation_gate_v10121.py
python .\run_verification_aware_batch_learning_v10120.py
python .\run_post_batch_internalized_verification_override_v10120.py
