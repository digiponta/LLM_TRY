# LLM_TRY v10.12.1 — Batch Repair Preservation Gate

v10.12.1 protects previously internalized knowledge during verification-aware batch repair.

## Motivation

v10.12.0 proved that multiple unstable internalized concepts can be repaired in one /trainbatch run and individually re-verified.

The remaining risk is catastrophic regression: repairing the selected concepts may damage other trusted INTERNALIZED knowledge.

v10.12.1 therefore adds a preservation gate after batch training.

## Protected set

Before /trainbatch starts:

protected set =
  checkpoint-bound trusted INTERNALIZED pairs
  minus current repair-target fingerprints

This baseline is fixed before training.

## Post-training preservation gate

After online_train.py finishes and the checkpoint is reloaded, each protected teacher question is sent directly to model weights using deterministic greedy generation.

The generated answer is evaluated with the same Composite Teacher Fidelity metrics used for repair verification:

- semantic similarity
- lexical coverage
- required-content coverage
- contradiction detection

A malformed/repetitive answer also fails.

## Batch outcome

All protected probes PASS:

[batch preservation gate: protected=N, passed=N, failed=0, status=PASS]

One or more protected probes FAIL:

[batch preservation gate: protected=N, passed=M, failed=K, status=FAIL]

The repaired checkpoint is still available for diagnosis, but the runtime explicitly reports that the batch must not be considered fully successful.

## Audit log

data/batch_preservation_v10121.jsonl

Each batch audit records:

- repair fingerprints
- protected count
- passed/failed counts
- per-concept generated answer
- semantic / lexical / required scores
- contradiction flag
- pass/fail reason
- timestamp

## Scope

The preservation gate evaluates model-internalized knowledge directly, even if normal runtime routing would prefer TYPED, CANONICAL, or UNIFIED retrieval.

Retrieval knowledge is not itself modified by online training and is outside this model-weight preservation gate.

## Regression

Run:

python .\run_batch_repair_preservation_gate_v10121.py

Expected:

STATUS : BATCH_REPAIR_PRESERVATION_GATE_PASS

Also rerun:

python .\run_verification_aware_batch_learning_v10120.py
python .\run_post_batch_internalized_verification_override_v10120.py
python .\run_internalized_verification_loop_v10119.py
python .\run_composite_teacher_fidelity_gate_v10118.py
