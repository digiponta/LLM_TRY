# LLM_TRY v10.12.12 — Corpus-to-Semantic Knowledge Training

v10.12.12 converts source-grounded knowledge in data-nagato.txt into explicit semantic propositions and trains them as structured knowledge rather than relying only on raw next-token continued pretraining.

## Pipeline

data-nagato.txt
-> source-grounded Xは... extraction
-> subject / relation / object_description
-> concept-disjoint TRAIN / HOLDOUT split
-> role balancing
-> preservation replay
-> semantic QA training
-> plain QA + semantic-role evaluation

## Why

Recent raw continued pretraining still improved held-out corpus NLL, but the incremental gain had fallen to roughly +0.5% and QA gain remained below threshold.

v10.12.12 therefore moves the corpus into the semantic training path already validated by v10.12.9 through v10.12.11.

## Dataset

Build:

python .\build_corpus_semantic_v101212.py

Outputs:

- data/nagato_corpus_semantic_train_v101212.jsonl
- data/nagato_corpus_semantic_holdout_seen_v101212.jsonl
- data/nagato_corpus_semantic_holdout_unseen_v101212.jsonl

Properties:

- one source-grounded proposition per concept
- no concept overlap between TRAIN and HOLDOUT
- comparison reserved as an unseen relation
- relations absent from TRAIN are moved to unseen-role HOLDOUT

## Training

python .\train_corpus_semantic_v101212.py

Candidate:

model/model-gpu-v1.6.2-online-corpus-semantic-candidate.pt

Training includes:

- deterministic role balancing
- three query forms per semantic proposition
- checkpoint-bound INTERNALIZED preservation replay
- first two transformer blocks frozen

## Evaluation

python .\evaluate_corpus_semantic_v101212.py

The evaluator compares production vs candidate on concept-disjoint HOLDOUT data using:

- plain QA gain
- semantic-role query gain
- seen-role transfer
- unseen-role transfer
- protected INTERNALIZED retention
- persona retention

Default generalization thresholds:

- mean plain QA gain >= +0.005
- mean semantic gain >= +0.010
- semantic improved count > regressed count

Final PASS additionally requires retention, persona, and v10.12.12 metadata.

## Full verification

python .\verify_corpus_semantic_v101212.py

Expected:

STATUS : CORPUS_TO_SEMANTIC_FULL_PASS

Do not promote the candidate until the full evaluation passes.
