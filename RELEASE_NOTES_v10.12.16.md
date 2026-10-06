# LLM_TRY v10.12.16 — Subject-Keyed Corpus Memory

v10.12.16 makes the canonical mapping

subject => full proposition

a persistent runtime knowledge source instead of relying on SFT alone.

## Motivation

v10.12.15 and v10.12.15.1 showed that explicit Subject-to-Proposition training can damage semantic generalization when mixed too strongly into the same model weights.

The mapping itself remains useful. The problem is where it is stored.

v10.12.16 therefore separates:

- corpus memory / retrieval
- semantic generalization in model weights

## Corpus memory

New persistent store:

data/subject_keyed_corpus_memory_v101216.jsonl

Canonical record:

GPU => GPUは高速である。

The store preserves:

- subject
- full proposition
- relation
- object_description
- source_text
- source

It is built from the complete source-grounded data-nagato corpus.

## Runtime resolution

Function resolution now prefers exact Subject-Keyed Corpus Memory evidence.

subject
-> exact corpus-memory lookup
-> full proposition evidence
-> action / target / purpose extraction
-> Pass-2 function composition

Generated multi-probe evidence remains available as secondary evidence.

No HOLDOUT teacher answer is injected dynamically.

## Important evaluation distinction

Corpus-memory retrieval and model generalization are evaluated separately.

A memory hit proves retrieval from the source corpus.

It does NOT count as unseen model generalization.

## Training

Some repeated learning is retained, but semantic learning remains dominant.

Defaults:

- semantic repeat: 2
- subject mapping weight: 1
- learning rate: 3e-6
- epochs: 4
- first two blocks frozen
- INTERNALIZED preservation unchanged

This produces repeated semantic-role exposure while keeping Subject-to-Proposition as an auxiliary objective.

## Verification

Run:

python .\verify_subject_keyed_corpus_memory_v101216.py

The full verification checks:

1. corpus-memory regression
2. corpus-memory build
3. exact retrieval coverage
4. semantic dataset build
5. repeated semantic training
6. semantic generalization
7. memory-backed function resolution

Expected final status:

STATUS : SUBJECT_KEYED_CORPUS_MEMORY_FULL_PASS

The candidate model is not promoted automatically.
