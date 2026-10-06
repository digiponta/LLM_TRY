# LLM_TRY v10.11.8 — Composite Teacher Fidelity Gate

v10.11.8 replaces single-score teacher fidelity with a composite gate for INTERNALIZED model generation.

## Motivation

v10.11.7 observed an incorrect internalized answer with teacher semantic cosine 0.938.
High cosine alone was therefore insufficient to prove that the generated answer preserved the trusted teaching content.

## Composite fidelity

An INTERNALIZED answer is evaluated using four signals:

1. semantic similarity
2. teacher lexical/content coverage
3. required-content term coverage
4. polarity contradiction detection

Default thresholds:

semantic >= 0.90
lexical coverage >= 0.45
required-content coverage >= 0.50
contradiction = false

All conditions are required.

## Required content

Required content is extracted from the trusted teacher answer after removing the concept subject and common grammatical glue.
The runtime records matched and missing terms so an unstable answer can be explained.

Example teacher:

量子センサーは、量子的性質を利用して高感度計測を行うセンサーである。

Observed bad candidate:

量子センサーは、量子的な役割と研究を学習して研究を行う研究を行う。

Even when semantic similarity remains high, lexical and required-content coverage can reject the candidate.

## Contradiction check

A simple polarity check detects cases where an affirmative trusted answer is converted to a negative claim, or vice versa.

Example:

teacher: 高感度計測を行う
candidate: 高感度計測を行わない

-> contradiction detected -> reject

## Runtime route

Composite failure:

resolution=INTERNALIZED_UNSTABLE
action=retrain/review
route=internalized fidelity block
AI=未学習です

INTERNALIZED generation still cannot be rescued solely by semantic probe agreement.

## Diagnostics

Runtime diagnostics include:

teacher_sem
teacher_lex
teacher_req
teacher_contra
fidelity_th
lex_th
req_th
missing

## Regression

python .\run_composite_teacher_fidelity_gate_v10118.py

Expected:

STATUS : COMPOSITE_TEACHER_FIDELITY_GATE_PASS

Also rerun:

python .\run_internalized_fidelity_gate_v10117.py
python .\run_internalized_checkpoint_binding_v10116.py
