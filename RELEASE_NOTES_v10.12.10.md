# LLM_TRY v10.12.10 — Semantic Role Generalization

v10.12.10 extends v10.12.9 from a single relation type (definition) to multiple semantic roles.

## Goal

Test two different kinds of transfer:

A. Seen-role / unseen-concept
- relation type is present in TRAIN
- subject/concept is unseen

B. Unseen-role
- relation type itself is excluded from TRAIN

This separates concept generalization from role-structure generalization.

## Supported semantic roles

The initial heuristic role extractor uses:

- definition
- property
- function
- cause
- comparison

Role inference is source-grounded and derived from the corpus sentence itself.

## Build dataset

python .\build_semantic_role_split_v101210.py

Default unseen relation:

comparison

Outputs:

- data/nagato_role_train_v101210.jsonl
- data/nagato_role_holdout_seen_v101210.jsonl
- data/nagato_role_holdout_unseen_v101210.jsonl

The builder fails closed if:

- subject leakage is detected
- unseen relation appears in TRAIN
- any split becomes empty

## Train

python .\train_semantic_role_generalization_v101210.py

Default candidate:

model/model-gpu-v1.6.2-online-semantic-role-candidate.pt

Each TRAIN proposition is presented in three forms:

1. original question
2. semantic role prompt
3. compact subject/relation prompt

## Evaluate

python .\evaluate_semantic_role_generalization_v101210.py

Seen-role PASS:

- mean gain >= +0.01
- improved > regressed

Unseen-role PASS:

- mean gain >= 0.00
- improved >= regressed

Final PASS additionally requires:

- protected INTERNALIZED retention
- persona retention
- semantic_role_version == v10.12.10

## Full verification

python .\verify_semantic_role_v101210.py

Expected final status:

SEMANTIC_ROLE_FULL_VERIFICATION_PASS

## Interpretation

If only seen-role passes:
- relation-aware learning transfers across unseen concepts
- unseen relation structure is not yet generalized

If both seen-role and unseen-role pass:
- the model shows evidence that subject/relation/object structure transfers beyond the trained relation inventory

Do not promote the candidate until the full evaluator passes.
