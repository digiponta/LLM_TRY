# LLM_TRY v10.12.12.2 — Raw+Semantic Candidate Promotion Gate

v10.12.12.2 adds a promotion gate dedicated to the final candidate produced by the v10.12.12.1 data-nagato full learning pipeline.

## Candidate

model/model-gpu-v1.6.2-online-nagato-semantic-candidate.pt

## Why a dedicated gate

The older v10.12.5 Nagato promotion gate validates the raw continued-pretraining candidate.

The v10.12.12.1 final candidate contains both:

- raw data-nagato continued pretraining
- corpus-to-semantic role-aware training

Therefore promotion must also verify semantic QA/generalization properties.

## Promotion gates

The new gate recomputes all important checks at promotion time.

It does not trust a stale PASS report.

Required:

- production trained-fingerprint binding is preserved
- Nagato raw-training metadata is present
- corpus_semantic_version == v10.12.12
- checkpoint-bound INTERNALIZED knowledge passes Composite Fidelity
- persona remains 長門有希
- mean plain QA gain >= +0.005
- mean semantic gain >= +0.010
- semantic improved > regressed
- seen-role semantic gain >= +0.010
- unseen-role semantic gain >= 0
- seen/unseen outcome policies pass

Audit:

data/raw_semantic_candidate_gate_v1012122.jsonl

## Evaluation only

python .\promote_raw_semantic_candidate_v1012122.py

A PASS leaves production unchanged.

## Promotion

python .\promote_raw_semantic_candidate_v1012122.py --promote

Promotion is atomic. A temporary production backup is restored if the replacement fails.

The candidate is moved into:

model/model-gpu-v1.6.2-online.pt

only after every gate passes.

## Full verification

python .\verify_raw_semantic_promotion_v1012122.py

Expected:

STATUS : RAW_SEMANTIC_PROMOTION_GATE_FULL_PASS
