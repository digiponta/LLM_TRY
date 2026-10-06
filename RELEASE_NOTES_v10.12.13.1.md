# LLM_TRY v10.12.13.1 — Preservation-Balanced Function Semantic Training

v10.12.13 proved that function decomposition into action / target / purpose is useful, but the training mixture was too aggressive.

Observed v10.12.13 result:

- function structured gain: +0.034 PASS
- generic function semantic gain: -0.012 FAIL
- non-function guard: -0.010 FAIL
- retention/persona/metadata: PASS

This means the new structured representation learned successfully, while older semantic behavior drifted.

## Fix

v10.12.13.1 changes the training mixture rather than abandoning the decomposition.

Defaults:

- learning rate: 4e-6 -> 2e-6
- epochs: 6 -> 4
- structured function weight: 4 -> 2
- generic function replay weight: 3
- non-function semantic replay weight: 2
- INTERNALIZED preservation replay: unchanged

Each function proposition now receives:

- established plain / semantic / compact prompts
- repeated generic semantic + compact prompts
- structured action / target / purpose prompts

Each non-function proposition receives extra semantic + compact replay.

The generic function replay is intentionally at least as large as structured replay.

## Verification

Run:

python .\verify_function_semantic_v1012131.py

Required evaluator conditions remain strict:

- generic function semantic mean gain >= +0.005
- structured function mean gain >= +0.010
- non-function guard >= -0.005
- improved > regressed for function semantic and structured prompts
- INTERNALIZED retention PASS
- persona PASS
- metadata PASS

Expected:

STATUS : FUNCTION_PRESERVATION_BALANCED_FULL_PASS

Candidate remains:

model/model-gpu-v1.6.2-online-function-semantic-candidate.pt

No automatic promotion is performed.
