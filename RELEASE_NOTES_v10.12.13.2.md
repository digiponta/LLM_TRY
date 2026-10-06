# LLM_TRY v10.12.13.2 — Leakage-Free Function Inference

v10.12.13.2 corrects an evaluation flaw discovered after v10.12.13 and v10.12.13.1.

## Finding

The previous structured-function HOLDOUT evaluation constructed:

action / target / purpose

from the HOLDOUT teacher answer itself.

Therefore a positive structured gain proved that the model could use an explicitly supplied function structure, but it did not prove that the model could infer that structure for an unseen concept.

This is teacher leakage.

## New leakage-free route

For HOLDOUT evaluation, the model now receives only:

subject
relation=function
action=?
target=?
purpose=?

Example:

主語: CPU
意味役割: function
action: ?
target: ?
purpose: ?
要求: action・target・purpose を学習済み知識から推定し、対象の機能を簡潔に説明してください

No HOLDOUT teacher answer or teacher-derived slot is present in the inference prompt.

## Training

Training may still use gold function decomposition for TRAIN examples as supervised auxiliary structure.

In addition, TRAIN function examples now include leakage-free inference prompts with hidden slots so the model is explicitly trained to infer function structure.

Defaults:

- learning rate: 1e-6
- epochs: 4
- function structured weight: 1
- function inference weight: 4
- generic function replay weight: 3
- non-function replay weight: 2
- INTERNALIZED preservation weight: 4

## Evaluation policy

Primary v10.12.13.2 gate:

- leakage-free function inference gain >= +0.010
- inference improved > regressed
- generic function prompt is compatibility guard >= -0.030
- oracle structured gain >= +0.010 (diagnostic/supporting)
- non-function semantic guard >= -0.005
- INTERNALIZED retention PASS
- persona PASS
- metadata PASS

The oracle structured metric is no longer considered evidence of unseen function generalization by itself.

## Verification

Run:

python .\verify_function_semantic_v1012132.py

Expected:

STATUS : LEAKAGE_FREE_FUNCTION_FULL_PASS

Candidate:

model/model-gpu-v1.6.2-online-function-semantic-candidate.pt

No automatic promotion is performed.
