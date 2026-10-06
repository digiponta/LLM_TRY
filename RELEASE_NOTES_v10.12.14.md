# LLM_TRY v10.12.14 — Two-Pass Function Semantic Resolver

v10.12.14 moves the remaining function-semantic problem from weight training to semantic routing and composition.

## Motivation

v10.12.13 showed that explicit action / target / purpose structure can help function answers.

v10.12.13.2 then removed teacher leakage and showed that a single-pass model does not reliably infer those slots for unseen function concepts.

The remaining problem is therefore treated as a two-stage runtime resolution problem instead of another SFT problem.

## Architecture

Pass 1:

subject
-> ordinary production-model query
-> generated evidence

Function decomposition:

generated evidence
-> action
-> target
-> purpose

Pass 2:

subject + extracted slots + generated evidence
-> production model
-> concise function answer

No teacher answer, HOLDOUT answer, or gold slot is used at runtime.

## Safety properties

- no model retraining
- no production checkpoint mutation
- non-function relations remain on the existing route
- malformed Pass-2 output falls back conservatively to Pass-1 evidence
- evaluator compares two-pass output against ordinary plain QA from the same production model

## Runtime CLI

Example:

python .\resolve_function_v101214.py --subject シングルコアCPU

The command displays:

- Pass-1 generated evidence
- extracted action / target / purpose
- Pass-2 function answer

## Evaluation

Run:

python .\evaluate_two_pass_function_v101214.py

Default PASS conditions:

- mean two-pass gain >= +0.005
- improved function cases > regressed cases
- informative slot rate >= 50%
- all non-function HOLDOUT relations bypass the resolver

## Full verification

Run:

python .\verify_two_pass_function_v101214.py

Expected:

STATUS : TWO_PASS_FUNCTION_FULL_PASS

This version intentionally performs no training and no automatic promotion.
