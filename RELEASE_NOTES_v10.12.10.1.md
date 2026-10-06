# LLM_TRY v10.12.10.1 — Preservation-Aware Semantic Role Training

v10.12.10.1 fixes two issues found in the v10.12.10 runtime experiment.

## Observed v10.12.10 results

- Seen-role mean gain: +0.009, just below the +0.010 threshold
- Unseen-role mean gain: +0.024, PASS
- Existing knowledge retention: FAIL because 文学 degraded
- Cause examples appeared in the seen-role holdout even though cause was absent from TRAIN

## Fix 1: Strict seen-role semantics

The dataset builder now uses a two-pass split.

A row is allowed into the seen-role / unseen-concept holdout only when its relation type actually exists in TRAIN.

Any provisional holdout row whose relation is absent from TRAIN is automatically moved to the unseen-role holdout.

This prevents unseen roles such as cause from being mislabeled as seen roles.

## Fix 2: Protected INTERNALIZED replay

Semantic-role training now reads checkpoint-bound protected INTERNALIZED records from:

data/chat_history.jsonl

The default replay multiplier is:

--preservation-weight 2

For the current five protected concepts this contributes 10 preservation rows:

- 宇宙
- 数学
- 文学
- 架空装置
- 量子センサー

These rows are mixed with semantic-role QA training data.

The goal is to retain verified knowledge while learning cross-role structure.

## Train

python .\train_semantic_role_generalization_v101210.py

The candidate remains:

model/model-gpu-v1.6.2-online-semantic-role-candidate.pt

Checkpoint metadata now contains:

semantic_role_version = v10.12.10.1

and records the protected concepts, replay weight, role rows, and preservation rows.

## Evaluate

python .\evaluate_semantic_role_generalization_v101210.py

The final PASS still requires:

- Seen-role mean gain >= +0.010
- Seen-role improved > regressed
- Unseen-role mean gain >= 0
- Unseen-role improved >= regressed
- protected INTERNALIZED retention PASS
- persona PASS
- v10.12.10.1 metadata PASS

## Regression

python .\run_preservation_aware_semantic_role_v1012101.py

Expected:

STATUS : PRESERVATION_AWARE_SEMANTIC_ROLE_PASS

## Full verification

Use the existing end-to-end runner:

python .\verify_semantic_role_v101210.py

It will rebuild the corrected split, retrain the preservation-aware candidate, and reevaluate both role-transfer axes.
