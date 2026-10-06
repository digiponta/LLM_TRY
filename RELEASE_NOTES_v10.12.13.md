# LLM_TRY v10.12.13 — Function Semantic Decomposition

v10.12.13 targets the remaining weak semantic relation observed after v10.12.12: function.

## Motivation

The previous corpus-to-semantic experiment passed overall generalization, but function remained unstable. In particular, examples such as シングルコアCPU could improve in plain QA while regressing under the generic semantic function prompt.

The hypothesis is that a single relation label:

function

is too coarse.

## Function decomposition

Function propositions are decomposed into:

- subject
- action
- target
- purpose

Example:

subject = CPU
action = execute
target = 命令
purpose = computation

The first implementation uses deterministic corpus-grounded heuristics for Japanese function expressions such as:

- 実行
- 処理
- 利用
- 使用
- 動作
- 行う
- 仲介

## Training

Run:

python .\train_function_semantic_v101213.py

Candidate:

model/model-gpu-v1.6.2-online-function-semantic-candidate.pt

Training keeps the existing semantic proposition prompts and adds function-specific structural prompts.

Default function emphasis:

--function-weight 4

Protected INTERNALIZED knowledge replay remains enabled.

## Evaluation

Run:

python .\evaluate_function_semantic_v101213.py

Required:

- old generic function semantic mean gain >= +0.005
- new structured function mean gain >= +0.010
- improved > regressed for both
- non-function semantic mean gain >= -0.005
- protected INTERNALIZED retention PASS
- persona PASS
- function_semantic_version == v10.12.13

The non-function guard prevents fixing function at the cost of other semantic relations.

## Full verification

python .\verify_function_semantic_v101213.py

Expected:

STATUS : FUNCTION_SEMANTIC_FULL_PASS

The candidate is not promoted automatically.
