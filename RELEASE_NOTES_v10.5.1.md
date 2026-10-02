# LLM_TRY v10.5.1 — Full Regression / Stable Candidate

## Purpose

v10.5.1 does not add another learning feature. It consolidates the major
regressions accumulated from v9.6 through v10.5 so the current adaptive model
can be evaluated as one stable candidate.

## Included regression areas

- known false-rejection preservation
- integrated Known / Unknown routing
- multi-turn history contamination
- Unknown -> Teaching control loop
- trained concept promotion
- concept-query-form promotion
- single-pair incremental training gate
- persistent adaptive checkpoint selection
- multi-concept incremental learning
- bare concept Known / Unknown gate

## Run

```powershell
python run_full_regression_v1051.py
```

Expected final summary:

```text
Passed            : 10/10
Failed            : 0/10
Stable candidate : PASS
```

If all ten suites pass, v10.5.1 can be treated as the current stable candidate
for the LLM_TRY experimental line.
