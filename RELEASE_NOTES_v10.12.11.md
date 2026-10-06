# LLM_TRY v10.12.11 — Role-Balanced Semantic Training

v10.12.11 builds on the successful v10.12.10.1 preservation-aware semantic-role experiment.

## Motivation

v10.12.10.1 passed:

- seen-role / unseen-concept generalization
- unseen-role generalization
- protected INTERNALIZED retention
- persona retention

However, the TRAIN relation distribution remained imbalanced, with property dominating and function remaining relatively weak.

## Role balancing

The trainer now counts source propositions per relation and deterministically oversamples minority roles toward the largest relation count.

Default:

--role-balance
--max-role-multiplier 4

Balancing is performed at the proposition level before the three prompt variants are generated.

This avoids over-counting prompt variants and keeps balancing tied to semantic role frequency.

## Preservation

v10.12.10.1 protection remains enabled.

Default:

--preservation-weight 2

Protected checkpoint-bound INTERNALIZED knowledge is replayed after semantic-role balancing.

## Training output

The existing candidate path is retained:

model/model-gpu-v1.6.2-online-semantic-role-candidate.pt

New checkpoint metadata includes:

- semantic_role_version = v10.12.11
- semantic_role_balance_enabled
- semantic_role_counts_before
- semantic_role_counts_after
- semantic_role_max_multiplier
- preservation metadata inherited from v10.12.10.1

## Evaluation

The evaluator continues to require:

Seen-role:
- mean gain >= +0.010
- improved > regressed

Unseen-role:
- mean gain >= 0
- improved >= regressed

Final PASS also requires:

- protected INTERNALIZED retention PASS
- persona PASS
- v10.12.11 metadata PASS

The evaluator now additionally prints per-relation gain summaries.

This makes function/property/etc. behavior visible instead of relying only on aggregate gain.

## Regression

Run:

python .\run_role_balanced_semantic_v101211.py

Expected:

STATUS : ROLE_BALANCED_SEMANTIC_TRAINING_PASS

## Full verification

Run:

python .\verify_role_balanced_semantic_v101211.py

Expected final status:

STATUS : ROLE_BALANCED_SEMANTIC_FULL_PASS
