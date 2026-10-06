# LLM_TRY v10.12.15.1 — Balanced Subject-to-Proposition Training

v10.12.15 successfully formalized the mapping:

subject => full proposition

but its first training mixture was too aggressive.

Observed v10.12.15:

- subject mappings: 56
- TRAIN subjects: 36
- subject mapping rows: 216
- total train rows: 379
- mean plain QA gain: -0.019
- mean semantic gain: -0.014
- semantic outcomes: improved 1 / regressed 19
- retention: PASS
- persona: PASS
- metadata: FAIL because the evaluator still expected v10.12.12

## Diagnosis

The failure does not invalidate Subject-to-Proposition itself.

The problem was the training ratio:

36 subjects × 2 mapping prompt forms × weight 3 = 216 mapping rows.

This overwhelmed the semantic-role objective.

The second prompt form also supplied the complete target proposition in the input:

GPU => GPUは高速である。

which mostly teaches copying rather than subject-keyed retrieval.

## v10.12.15.1 correction

The formal dataset field remains:

subject_mapping = "GPU => GPUは高速である。"

Training now uses only the lookup form:

GPU =>

with target:

GPUは高速である。

Default mapping weight is reduced:

3 -> 1

For 36 TRAIN subjects this produces only 36 mapping rows instead of 216.

Semantic-role training and INTERNALIZED preservation remain unchanged.

## Metadata

The evaluator now requires:

- corpus_semantic_version == v10.12.15.1
- subject_to_proposition_version == v10.12.15.1
- subject_mapping_rows > 0

## Goal

Subject-to-Proposition is now an auxiliary retrieval objective rather than the dominant objective.

The intended hierarchy is:

semantic-role learning
+ subject-keyed full-proposition retrieval
+ protected knowledge replay

## Verification

Run:

python .\verify_subject_to_proposition_v101215.py

Expected:

STATUS : SUBJECT_TO_PROPOSITION_FULL_PASS
