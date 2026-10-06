# LLM_TRY v10.12.0 — Verification-Aware Batch Learning

v10.12.0 extends the v10.11.9 single-item Internalized Verification Loop into multi-task batch repair.

## Motivation

v10.11.9 demonstrated:

INTERNALIZED_UNSTABLE
→ trusted-pair reactivation
→ /train
→ re-query
→ verified

That flow worked for one concept at a time.

v10.12.0 generalizes the same repair policy to multiple unstable internalized concepts.

## Batch repair plan

All active verification tasks in:

data/internalized_verification_v10119.jsonl

with status:

- pending
- retrain

are collected into one BatchRepairPlan.

The plan records:

- active tasks
- unique trusted fingerprints
- unique concepts

## New commands

### /batchstatus

Shows the current verification-aware repair batch.

Example:

[verification batch: tasks=3, fingerprints=3, concepts=3]
  01. concept='量子センサー' status=retrain ...
  02. concept='宇宙' status=retrain ...
  03. concept='数学' status=retrain ...

### /trainbatch

Performs:

1. collect all active verification tasks
2. reactivate all associated trusted pairs in chat_learning_state.json
3. mark the tasks as retrain
4. invoke online_train.py once
5. reload the resulting checkpoint
6. refresh checkpoint binding
7. print the concepts that require post-training re-verification

The actual model training still uses the existing online_train.py multi-pending-row path.

## Why reuse online_train.py

online_train.py already supports multiple pending trusted rows in one training run:

- New/rebind = N
- Prior trusted = M
- weighted manual/recovery replay
- stability replay
- one output checkpoint

Therefore v10.12.0 adds orchestration rather than a second training implementation.

## Verification lifecycle

For multiple concepts:

unstable A
unstable B
unstable C
→ one batch repair plan
→ one /trainbatch
→ one updated checkpoint
→ re-query A/B/C
→ each successful concept independently transitions to verified

A failed concept remains active for another repair cycle.

## Safety

Training is not triggered automatically by generation failure.

The model may create verification tasks automatically, but the user must explicitly run:

/trainbatch

This prevents uncontrolled training loops.

## Regression

Run:

python .\run_verification_aware_batch_learning_v10120.py

Expected:

STATUS                : VERIFICATION_AWARE_BATCH_LEARNING_PASS

Also rerun:

python .\run_internalized_verification_loop_v10119.py
python .\run_composite_teacher_fidelity_gate_v10118.py
python .\run_internalized_fidelity_gate_v10117.py
python .\run_internalized_checkpoint_binding_v10116.py
