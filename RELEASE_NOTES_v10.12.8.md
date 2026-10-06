# LLM_TRY v10.12.8 — Corpus-to-QA Knowledge Internalization

v10.12.8 tests whether raw corpus knowledge can be converted into question-answerable knowledge without contaminating evaluation data.

## Pipeline

data/data-nagato.txt
  -> source-grounded QA extraction
  -> deterministic concept-level TRAIN / HOLDOUT split
  -> TRAIN-only lightweight QA SFT
  -> unseen HOLDOUT QA evaluation
  -> INTERNALIZED preservation
  -> persona preservation
  -> candidate acceptance / rejection

## 1. Build TRAIN / HOLDOUT QA data

Run:

python .\build_nagato_qa_split_v10128.py

Default outputs:

- data/nagato_qa_train_v10128.jsonl
- data/nagato_qa_holdout_v10128.jsonl

The split is deterministic by concept.

The v10.12.7 evaluation concepts are explicitly reserved for HOLDOUT when present, preventing benchmark leakage.

TRAIN and HOLDOUT concept overlap must be zero.

## 2. Train QA candidate

Run:

python .\train_nagato_qa_sft_v10128.py

Defaults:

- Base: model/model-gpu-v1.6.2-online.pt
- Output: model/model-gpu-v1.6.2-online-nagato-qa-candidate.pt
- Epochs: 6
- LR: 5e-6
- Batch size: 4
- Frozen transformer blocks: first 2

Only TRAIN QA data is accepted.

The trainer rejects any row whose split_reason contains "holdout".

Checkpoint metadata is inherited and augmented with:

corpus_to_qa_version = v10.12.8

## 3. Evaluate unseen HOLDOUT QA

Run:

python .\evaluate_nagato_qa_holdout_v10128.py

Default comparison:

Before:
model/model-gpu-v1.6.2-online.pt

After:
model/model-gpu-v1.6.2-online-nagato-qa-candidate.pt

HOLDOUT PASS requires:

- mean holdout composite QA gain >= +0.01
- improved holdout probes > regressed probes

The evaluator also requires:

- protected INTERNALIZED knowledge retention
- persona "長門有希。" retention
- v10.12.8 checkpoint metadata

Final output:

[corpus-to-QA summary: ..., status=PASS|FAIL]

## 4. Promotion policy

Do not promote the QA candidate unless the holdout evaluator returns PASS.

A later promotion helper can reuse the v10.12.5 transaction pattern once runtime results confirm the holdout policy.

## Regression

Run:

python .\run_corpus_to_qa_internalization_v10128.py

Expected:

STATUS : CORPUS_TO_QA_INTERNALIZATION_PASS
