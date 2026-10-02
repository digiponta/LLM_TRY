# LLM_TRY v10.1 Release Notes

## Summary

v10.1 promotes the v10.0 Known/Unknown gate architecture to a new stable
experimental baseline by fixing a multi-turn semantic-history contamination
bug.

The language model and canonical SFT checkpoint remain unchanged:

```text
model/model-llm-try-nagato-chat-v94.pt
```

The change is in chat-time semantic validation.

## Problem Found After v10.0

Interactive multi-turn testing exposed a mismatch between generation context
and semantic contamination checking.

Generation used:

```text
selected_history
```

but the semantic consistency checker used:

```text
full history
```

This caused a correct response to `AIとは` to be rejected after several
persona questions, even though those older turns were not part of the actual
generation prompt.

Observed failure:

```text
candidate:
データから規則性を学習し、推論や生成を行う計算システム。
目的に応じて学習方法や構成が変わる。

gate:
UNKNOWN

reason:
history contamination
```

## Fix

`chat.py` now passes `selected_history` to the semantic contamination check.

Final alignment:

```text
User query
    |
    v
build_prompt(...)
    |
    +-- selected_history --> generation
    |
    +-- selected_history --> semantic contamination check
```

This removes contamination from turns that were not actually supplied to the
current generation prompt.

## New Regression

New file:

```text
eval_multiturn_history_v101.py
```

Test sequence:

```text
あなたは誰ですか
名前を教えてください
自己紹介してください
AIとは
LLMって何
CUDAとは
量子力学とは
宇宙とは
```

Verified result:

```text
Passed            : 8/8
Failed            : 0/8
Regression status : PASS
```

## Existing Regression Status

The v10.0 regression baseline remains:

```text
Known preservation : 11/11 = 100.0%
Unknown rejection  : 20/20 = 100.0%
Balanced accuracy  : 100.0%
Parse failures     : 0
Regression status  : PASS
```

The known false-rejection diagnostic remains:

```text
PASS         : 11/11
FALSE_REJECT : 0/11
MODEL_FAIL   : 0/11
PARSE_FAIL   : 0/11
```

## Recommended Verification

```powershell
python eval_known_false_rejection_v96.py
python eval_integrated_gate_v97.py
python eval_multiturn_history_v101.py
```

## v10.1 Stable Architecture

```text
User Query
   |
   v
Pre-generation Concept Gate
   |
   +-- Unknown --------------------------> 未学習です
   |
   +-- Known
        |
        v
      LLM generation
        |
        +-- selected_history
        |
        v
Semantic / Category / Confidence Gate
        |
        +-- contamination check uses the same selected_history
        |
        +-- ACCEPT ----------------------> Answer
```

## Status

v10.1 is the recommended stable experimental baseline for continuing LLM_TRY
work on:

- larger held-out known/unknown sets
- semantic concept registration
- dynamic known-concept expansion
- incremental teaching
- broader multi-turn evaluation
