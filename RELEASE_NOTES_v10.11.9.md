# LLM_TRY v10.11.9 — Internalized Verification Loop

v10.11.9 turns INTERNALIZED_UNSTABLE from a terminal rejection into a managed verification/retraining loop.

## Motivation

v10.11.8 can correctly reject an internalized answer when composite teacher fidelity fails.
The next step is to recover automatically from that unstable state instead of only reporting the failure.

## Verification loop

INTERNALIZED=CURRENT
  -> model generation
  -> Composite Teacher Fidelity Gate
  -> PASS: answer accepted; matching verification task resolved
  -> FAIL: INTERNALIZED_UNSTABLE
       -> persist teacher/candidate difference
       -> reactivate trusted pair in chat_learning_state.json
       -> mark verification task as retrain
       -> /train
       -> reload checkpoint binding
       -> query again
       -> PASS resolves task as verified

## Persistent store

data/internalized_verification_v10119.jsonl

Each task records:

- concept
- trusted question
- teacher answer
- generated candidate
- composite fidelity reason
- semantic score
- lexical coverage
- required-content coverage
- contradiction flag
- missing required terms
- trusted-pair fingerprint
- status
- attempts
- timestamps

Lifecycle:

pending -> retrain -> verified

## Runtime behavior

When INTERNALIZED_UNSTABLE is detected, the trusted teacher pair is removed from the current learning-state trained set so online_train.py treats it as a rebind/retraining target.
The checkpoint itself is not immediately modified. /train performs the actual update.

This avoids an uncontrolled automatic training loop.

## Commands

/verifystatus
  Show pending / retrain / verified / failed counts.

/verifications
  List active pending/retrain tasks with fidelity scores and missing terms.

## Diagnostics

Composite fidelity diagnostics are now consistently displayed as:

teacher_sem
teacher_lex
teacher_req
teacher_contra
fidelity_th
lex_th
req_th
missing

## Regression

python .\run_internalized_verification_loop_v10119.py

Expected:

STATUS : INTERNALIZED_VERIFICATION_LOOP_PASS

Also rerun:

python .\run_composite_teacher_fidelity_gate_v10118.py
python .\run_internalized_fidelity_gate_v10117.py
python .\run_internalized_checkpoint_binding_v10116.py
