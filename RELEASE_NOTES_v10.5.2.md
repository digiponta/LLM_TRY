# LLM_TRY v10.5.2 — Stable Adaptive Learning Release

## Status

Stable release candidate verified with the consolidated regression suite:

~~~text
Passed            : 10/10
Failed            : 0/10
Stable candidate : PASS
~~~

## What v10.5.2 establishes

- pre-generation Known / Unknown routing;
- explicit human teaching through `/teach` and `/teachq`;
- single-pair GPU incremental training;
- trained-concept promotion only after the trusted pair is actually consumed;
- persistent adaptive checkpoint resume after process restart;
- multi-concept incremental learning;
- Stability Replay to reduce catastrophic forgetting;
- quality-aware adaptive checkpoint selection using strict pass + teacher-answer fidelity;
- Bare Concept Gate for short concept-only inputs;
- adaptive-state-aware regression expectations.

## Verified preserved knowledge

~~~text
Nagato persona
AI
LLM
CUDA
量子力学
~~~

## Verified incrementally learned knowledge

~~~text
宇宙
数学
文学
~~~

## Verified unknown isolation

~~~text
ブラックホール
相対性理論
化学
~~~

## Adaptive checkpoint selection

The v10.4.3 quality-aware sweep selected a strict-pass candidate with:

~~~text
Manual weight    : 6
Stability weight : 2
Learning rate    : 1e-5
Score            : 11/11
Strict pass      : True
Learned fidelity : 1.000
~~~

## Full regression

Run:

~~~powershell
python run_full_regression_v1051.py
~~~

Verified suites:

~~~text
[PASS] known-false-rejection-v96
[PASS] integrated-known-unknown-v97
[PASS] multiturn-history-v101
[PASS] unknown-teaching-loop-v102
[PASS] trained-concept-promotion-v1021
[PASS] concept-query-promotion-v1022
[PASS] single-pair-training-gate-v1023
[PASS] persistent-checkpoint-v1024
[PASS] multiconcept-incremental-v104
[PASS] bare-concept-gate-v105
~~~

## Current adaptive learning loop

~~~text
UNKNOWN
  -> knowledge queue
  -> explicit teaching
  -> incremental training
  -> trained fingerprint
  -> concept promotion
  -> persistent adaptive checkpoint
  -> restart-safe KNOWN routing
~~~

## Notes

This release is an experimental stable baseline for the LLM_TRY research line.
The 10/10 result is a controlled regression result for the included scenarios,
not a claim of general-purpose model accuracy.

The canonical v9.4 checkpoint remains the stable base model. Adaptive knowledge
is carried by `model/model-gpu-v1.6.2-online.pt` and
`data/chat_learning_state.json`.
