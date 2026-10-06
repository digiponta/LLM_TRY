# LLM_TRY v10.11.3 — Bare Unknown Safety Gate

v10.11.3 prevents unsupported bare concepts from reaching free model
generation.

## Motivation

Observed before v10.11.3:

~~~text
You> 時間
AI> 恩せられて。
...
reason=semantic probe agreement
~~~

The word "時間" existed only in raw corpus evidence. It was intentionally not
promoted by v10.11.2, but then fell through to normal generation where semantic
probe agreement could incorrectly accept a poor answer.

## New policy

~~~text
Bare input
   ↓
Validated Semantic Knowledge?
   ├─ yes
   │   -> canonicalize to Xとは
   │   -> Semantic route
   │
   └─ no
       ↓
   Conversational allowlist?
       ├─ yes -> normal generation
       └─ no  -> UNKNOWN / BLOCK
~~~

Validated semantic knowledge includes:

~~~text
Subject / Typed Proposition
Canonical Definition
Unified Semantic Memory
Internalized Knowledge
~~~

Raw corpus occurrence is not sufficient.

## Conversational preservation

The initial conservative allowlist preserves:

~~~text
長門
長門有希
hello
hi
~~~

Thus:

~~~text
長門
  -> normal model generation
  -> persona behavior preserved
~~~

while:

~~~text
時間
  -> bare unknown safety gate
  -> UNKNOWN
  -> BLOCK
  -> 0 generated tokens
~~~

## Explicit query behavior

Explicit queries retain the existing knowledge-state resolver behavior.

Example:

~~~text
時間とは
  -> RAW_CORPUS_ONLY
  -> BLOCK
~~~

This is distinct from bare "時間", which is blocked earlier as an unsupported
bare concept.

## Result metadata

SemanticKnowledgeResult now includes:

~~~text
bare_focus
bare_concept_routed
bare_unknown_blocked
routed_query
~~~

## Runtime diagnostics

When blocked before generation:

~~~text
[bare unknown safety gate: concept='時間', generation=blocked]
~~~

The existing UNKNOWN_KNOWLEDGE teaching/retrieval path is then used.

## Regression

~~~powershell
python .\run_bare_unknown_safety_gate_v10113.py
~~~

Expected:

~~~text
STATUS : BARE_UNKNOWN_SAFETY_GATE_PASS
~~~
