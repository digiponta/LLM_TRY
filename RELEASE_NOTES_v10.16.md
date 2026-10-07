# LLM_TRY v10.16 — Modifier-to-Condition Normalization

## Purpose

v10.16 adds a conservative Japanese semantic surface normalization step:

~~~text
modifier + subject + predicate
        |
        v
subject + modifier + 場合 + predicate
~~~

Primary example:

~~~text
高温のCPUは停止する
    ->
CPUは、高温の場合、停止する
~~~

## Design rule

The normalizer converts only modifiers that are positively classified as
conditions/states. It intentionally does not rewrite ordinary attributes,
ownership, or semantic relations.

Converted examples:

~~~text
低温のCPUは正常に動作する
    -> CPUは、低温の場合、正常に動作する

高負荷のGPUは温度が上がる
    -> GPUは、高負荷の場合、温度が上がる

雨の日の道路は滑る
    -> 道路は、雨の日の場合、滑る
~~~

Preserved examples:

~~~text
赤い車は速い
日本の首都は東京である
文学の分類は複雑である
~~~

## Runtime integration

The v10.16 branch integrates the normalizer into `chat.py::normalize_runtime_input`.
Therefore accepted ordinary chat input is canonicalized before routing, semantic
lookup, generation, and learning capture.

The existing v10.13.x semantic/retrieval/truth architecture remains unchanged.

## Verification

Run:

~~~powershell
python .\verify_modifier_condition_v10160.py
~~~

Expected final status:

~~~text
STATUS : MODIFIER_TO_CONDITION_V10160_PASS
~~~

The regression covers positive conversions, attribute/relation preservation, and
idempotence of already-canonical input.
