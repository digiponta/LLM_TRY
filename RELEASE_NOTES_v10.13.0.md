# LLM_TRY v10.13.0 — Semantic Knowledge Runtime Stable Release

v10.13.0 is the stable runtime consolidation point for the LLM_TRY semantic knowledge experiments.

## Stable scope

This release freezes the runtime architecture validated through v10.12.16.1:

- Semantic Knowledge Architecture facade
- Atomic Proposition store
- Subject and Typed indexes
- Unified Semantic Memory
- Internalized Knowledge and checkpoint binding
- Provenance
- Truth-State overlay
- Knowledge-State Resolver / Dispatcher
- Subject-Keyed Corpus Memory
- Retrieval-First Runtime
- Function action / target / purpose extraction
- Unknown fallback and existing chat gates

## Key experimental conclusion

The project tested three ways to make data-nagato knowledge usable:

1. repeated raw continued pretraining
2. Subject-to-Proposition SFT
3. explicit Subject-Keyed Corpus Memory

The stable runtime selects the third as the primary source-grounded retrieval path.

Subject-to-Proposition remains a canonical representation:

subject => full proposition

but v10.13.0 does not require additional SFT to use it.

## Verified predecessor result

v10.12.16.1 produced:

- Subject HIT rate: 100.00%
- Proposition coverage: 100.00%
- Function slot rate: 100.00%
- Unknown fallback: PASS
- Generation required for memory HIT: NO
- Model retraining: NONE
- Production mutation: NONE

These are controlled regression measurements on the current data-nagato corpus, not general-purpose accuracy claims.

## Runtime priority

Concept query:

Subject-Keyed Corpus Memory HIT
-> full proposition retrieval
-> relation/function metadata
-> direct response

MISS
-> existing Semantic Knowledge Architecture
-> RETRIEVE / GENERATE / BLOCK

Explicit FALSE / OUTDATED / CONTESTED truth states bypass direct corpus-memory return and remain under the Truth-Aware path.

## Stable checkpoint

Production remains:

model/model-gpu-v1.6.2-online.pt

No v10.13.0 model retraining is performed.

## Verification

Run:

python .\verify_semantic_runtime_stable_v10130.py

Expected final result:

STATUS : SEMANTIC_KNOWLEDGE_RUNTIME_STABLE_FULL_PASS

The verifier does not train or promote a model.


## Verification dependency correction

The initial v10.13.0 verifier referenced the historical filename
run_chat_learning_regression_v1626.py, which is not present in LLM_TRY.

The stable verifier now uses the repository's existing consolidated regression:

run_full_regression_v1051.py

This correction changes only the verification dependency; it does not change
the v10.13.0 runtime architecture or production model.


## Stable regression contract update

A second verification issue was found after replacing the missing historical
script. The old v10.5 regression suite itself encodes pre-Retrieval-First
expectations, including model-known AI/LLM/quantum-mechanics behavior and
unknown treatment for concepts that are now source-grounded in corpus memory.

v10.13.0 therefore uses:

run_stable_chat_runtime_v10130.py

This validates the current stable contract directly:

- input-quality rejection
- corpus-memory retrieval
- unknown memory miss / fallback
- function structure extraction
- Truth-State direct-retrieval policy

Historical v10.5 regressions remain available as historical experiments but
are no longer a release gate for v10.13.0.
