# LLM_TRY v10.12.16.1 — Retrieval-First Runtime

v10.12.16.1 removes retraining from the runtime verification path and makes Subject-Keyed Corpus Memory the first knowledge source for concept queries.

## Runtime priority

For explicit or bare concept queries:

1. extract subject
2. exact lookup in data/subject_keyed_corpus_memory_v101216.jsonl
3. if HIT, return the stored full proposition directly
4. attach relation/function metadata
5. if MISS, continue through the existing Semantic Knowledge Architecture / model route

## Truth-state safety

Direct corpus-memory retrieval is bypassed when the explicit truth state is:

- FALSE
- OUTDATED
- CONTESTED

Those concepts continue through the existing Truth-Aware path.

UNVERIFIED corpus entries remain retrievable with provenance shown as corpus memory.

## Function handling

When the retrieved corpus relation is function, runtime also derives:

- action
- target
- purpose

from the retrieved full proposition.

This avoids relying on weak generated Pass-1 evidence.

## No retraining

The full v10.12.16.1 verification performs:

- corpus memory build
- retrieval-first regression
- retrieval-first corpus evaluation
- chat.py syntax validation

It intentionally performs no semantic retraining and does not mutate production.

## Chat integration

chat.py now supports:

--corpus-memory data/subject_keyed_corpus_memory_v101216.jsonl

A memory HIT returns with:

0 generated probe tokens, corpus-memory retrieval

## Verification

Run:

python .\verify_retrieval_first_runtime_v1012161.py

Expected:

STATUS : RETRIEVAL_FIRST_RUNTIME_FULL_PASS
