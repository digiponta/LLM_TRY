# LLM_TRY v10.12.3 — Candidate Repair Quality Gate

v10.12.3 strengthens candidate promotion by validating the repair targets themselves before promotion.

## Promotion rule

A candidate may be promoted only when both conditions pass:

- Preservation Gate PASS
- Repair Quality Gate PASS

Formally:

promotion = preservation_ok AND repair_quality_ok

If either gate fails, the candidate checkpoint and candidate learning state are discarded and the current production pair remains active.

## Repair Quality Gate

For every active batch repair task:

1. use the trusted repair question directly
2. generate from the candidate model with deterministic greedy decoding
3. compare candidate output with the trusted teacher answer
4. apply the Composite Teacher Fidelity Gate:
   - semantic similarity
   - lexical coverage
   - required-content coverage
   - contradiction detection
5. reject malformed or repetitive output
6. require every repair target to pass

Expected PASS:

[repair quality PASS: concept='文学', sem=..., lex=..., req=..., contra=False]
[repair quality PASS: concept='架空装置', sem=..., lex=..., req=..., contra=False]
[candidate repair quality gate: targets=2, passed=2, failed=0, status=PASS]

## Rollback behavior

If Preservation passes but one repair target fails:

[candidate repair quality gate: targets=2, passed=1, failed=1, status=FAIL]
[batch rollback: repair_quality=FAIL; candidate checkpoint/state discarded]
[production checkpoint retained: ...]

Verification tasks remain retrain so the repair can be attempted again with adjusted training/replay.

## Audit log

data/candidate_repair_quality_v10123.jsonl

Each audit contains:

- repair concept
- trusted question
- fingerprint
- generated candidate answer
- semantic score
- lexical score
- required-content score
- contradiction flag
- pass/fail reason
- aggregate batch result

## Runtime test hook

Use:

python .\chat.py --candidate-repair-force-fail

This forces the Repair Quality Gate to fail after real candidate evaluation, allowing safe runtime verification of the repair-quality rollback path.

## Regression

Run:

python .\run_candidate_repair_quality_gate_v10123.py

Expected:

STATUS : CANDIDATE_REPAIR_QUALITY_GATE_PASS

Also rerun:

python .\run_batch_repair_candidate_rollback_v10122.py
python .\run_batch_repair_preservation_gate_v10121.py
