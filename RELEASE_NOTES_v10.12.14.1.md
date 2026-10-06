# LLM_TRY v10.12.14.1 — Multi-Probe Two-Pass Function Resolver

v10.12.14 achieved positive two-pass QA gain but failed to extract any informative function slots.

Observed v10.12.14 result:

- mean two-pass gain: +0.010 PASS
- improved: 2 / regressed: 0
- informative slot rate: 0% FAIL
- non-function route guard: PASS
- teacher leakage: NONE
- weight update: NONE

## Diagnosis

The failure occurred before slot extraction.

The single Pass-1 query:

Xとは

did not reliably generate function-bearing verbs such as 実行 / 処理 / 利用 / 使用.

Therefore action / target / purpose remained:

function / unspecified / unspecified

even though the two-pass composition itself improved the answer.

## v10.12.14.1 change

Pass 1 now uses three independent leakage-free evidence probes:

- Xとは
- Xは何をするものですか
- Xの役割は何ですか

All usable generated answers are deduplicated and merged.

Only this model-generated evidence is passed to the deterministic function slot extractor.

No teacher answer or gold slot is used.

## Runtime

python .\resolve_function_v101214.py --subject シングルコアCPU

The CLI displays each evidence probe separately, the merged evidence, extracted slots, and final Pass-2 answer.

## Evaluation

python .\evaluate_two_pass_function_v101214.py

PASS conditions remain:

- mean two-pass gain >= +0.005
- improved > regressed
- informative slot rate >= 50%
- non-function route guard PASS
- model retraining NONE
- teacher leakage NONE

## Full verification

python .\verify_two_pass_function_v1012141.py

Expected:

STATUS : MULTI_PROBE_TWO_PASS_FUNCTION_FULL_PASS
