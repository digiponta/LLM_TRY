# LLM_TRY v10.12.9 — Semantic QA Generalization

v10.12.9 follows the v10.12.8 result:

- TRAIN fitting improved
- existing knowledge retention passed
- persona retention passed
- unseen HOLDOUT QA generalization failed

The new experiment tests whether an explicit shared semantic relation structure can improve transfer to unseen concepts.

## Semantic decomposition

Each source-grounded definition QA is represented as:

subject = concept
relation = definition
object_description = predicate / description from the corpus sentence

Example:

question:
物質波とは

answer:
物質波は、ドブロイ波と言われているもの。

structure:

subject = 物質波
relation = definition
object_description = ドブロイ波と言われているもの

## Relation-aware training

For each TRAIN concept, the trainer uses three query forms:

1. original plain question
2. semantic query with concept + relation
3. compact subject/relation query

All three map to the same corpus-grounded answer.

HOLDOUT concepts remain completely excluded from training.

## Train

python .\train_semantic_qa_generalization_v10129.py

Defaults:

- base: model/model-gpu-v1.6.2-online.pt
- output: model/model-gpu-v1.6.2-online-semantic-qa-candidate.pt
- epochs: 6
- learning rate: 5e-6
- frozen transformer blocks: first 2
- relation: definition

## Evaluate

python .\evaluate_semantic_qa_generalization_v10129.py

For each unseen HOLDOUT concept, evaluation measures:

- before plain score
- after plain score
- before semantic-query score
- after semantic-query score
- plain gain
- semantic gain
- semantic lift

Definitions:

semantic gain =
post-training semantic-query score
minus
pre-training semantic-query score

semantic lift =
post-training semantic-query score
minus
post-training plain-query score

Generalization PASS requires:

- mean semantic gain >= +0.01
- improved HOLDOUT concepts > regressed HOLDOUT concepts

Final PASS also requires:

- protected INTERNALIZED retention
- persona retention
- semantic_qa_version == v10.12.9

## Regression

python .\run_semantic_qa_generalization_v10129.py

Expected:

STATUS : SEMANTIC_QA_GENERALIZATION_PASS

## Promotion

Do not promote the semantic QA candidate until the unseen HOLDOUT evaluator returns PASS.
