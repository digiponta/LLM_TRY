# LLM_TRY v10.12.15 — Subject-to-Proposition Training

v10.12.15 makes Subject -> Full Proposition a formal training format for data-nagato.txt.

## Canonical mapping

Source sentence:

GPUは高速である。

is represented conceptually and in the generated dataset as:

GPU => GPUは高速である。

The right-hand side is always the full canonical proposition, including the subject.

## Dataset

build_corpus_semantic_v101212.py now stores:

- subject
- relation
- object_description
- question
- answer
- subject_mapping

Example:

subject_mapping = "GPU => GPUは高速である。"

Existing concept-disjoint TRAIN/HOLDOUT splitting remains unchanged.

## Training

train_corpus_semantic_v101212.py keeps the existing semantic-role training and additionally trains explicit subject lookup.

For each subject/proposition pair, two prompt forms are used:

- "GPU =>"
- "GPU => GPUは高速である。"

Both target the full proposition:

GPUは高速である。

Default subject mapping weight:

--subject-mapping-weight 3

Therefore the model is explicitly encouraged to recover full proposition evidence directly from a subject key.

## Motivation

The v10.12.14 series showed that function resolution can improve when usable subject evidence is available, but ordinary Xとは generation does not always expose the relevant predicate.

Subject-to-Proposition training strengthens:

subject
-> full proposition
-> semantic decomposition
-> relation/function resolution

without replacing the existing semantic-role representation.

## Verification

Run:

python .\verify_subject_to_proposition_v101215.py

Expected:

STATUS : SUBJECT_TO_PROPOSITION_FULL_PASS
