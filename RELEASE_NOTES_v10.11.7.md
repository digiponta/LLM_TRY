# LLM_TRY v10.11.7 — Internalized Fidelity Gate

v10.11.7 verifies that a CURRENT internalized checkpoint reproduces the trusted teacher meaning before accepting its generated answer.

## Motivation

Observed after v10.11.6 rebinding:

Teacher:
量子センサーは、量子的性質を利用して高感度計測を行うセンサーである。

Generated:
量子センサーは、量子的な役割と研究を学習して研究を行う研究を行う。

The ordinary output gate accepted this because probe semantic agreement was high even though the answer drifted from the trusted teaching target.

## Policy

INTERNALIZED=CURRENT
  -> generate
  -> normal output-quality / semantic checks
  -> teacher fidelity cosine
  -> accept only when fidelity >= threshold

Default threshold:

min_internalized_fidelity = 0.90

## Semantic rescue

For INTERNALIZED generation, semantic probe agreement no longer rescues low lexical agreement.
Probe consensus cannot prove fidelity to the teacher when all probes drift in the same direction.

## Failure route

When teacher fidelity fails:

resolution=INTERNALIZED_UNSTABLE
action=retrain/review
route=internalized fidelity block
AI=未学習です

The candidate is shown only as diagnostic output when risk display is enabled.

## Diagnostics

Gate output includes:

teacher_fidelity=<score>
fidelity_th=0.900

## Regression

python .\run_internalized_fidelity_gate_v10117.py

Expected:

STATUS : INTERNALIZED_FIDELITY_GATE_PASS

Also rerun:

python .\run_internalized_checkpoint_binding_v10116.py
