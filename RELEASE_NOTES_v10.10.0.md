# LLM_TRY v10.10.0 — Typed Semantic Proposition

v10.10.0 extends the v10.9.0 subject-keyed proposition architecture from:

~~~text
Subject => Statement
~~~

to:

~~~text
Subject => Predicate Type => Statement
~~~

The atomic proposition store remains the source of truth. Typed semantic data
is derived and can therefore be rebuilt without changing learned model weights.

## Predicate types

The initial controlled schema contains four predicate classes:

~~~text
definition
property
capability
relation
~~~

Examples:

~~~text
GPU => property   => GPUは高速である。
GPU => capability => GPUは並列計算が得意である。
GPU => relation   => GPUはCUDAを利用可能である。
GPU => definition => GPUは多数の演算を並列に実行する処理装置である。
~~~

Predicate classification is deterministic and auditable. It uses conservative
Japanese surface rules rather than an unrestricted learned classifier.

## Runtime pipeline

~~~text
Natural Language
      ↓
Atomic Proposition Store
      ↓
Subject-Keyed Index
      ↓
Typed Subject Index
      ↓
Unified Semantic Memory
      ↓
Runtime Retrieval
~~~

The composed natural-language answer remains unchanged by typing. Predicate
types are metadata used for routing, inspection, provenance and future semantic
operations.

## Unified semantic provenance

Rows synchronized from atomic propositions now carry typed metadata:

~~~json
{
  "concept": "GPU",
  "assistant": "GPUは、高速であり、並列計算が得意である。",
  "source": "atomic-proposition",
  "atomic_count": 2,
  "predicate_types": ["property", "capability"],
  "semantic_schema": "subject-predicate-type-statement-v10.10.0"
}
~~~

## Chat commands

~~~text
/typedsubject GPU
/typedsubjects
~~~

`/propteach` rebuilds both the untyped and typed indexes automatically.
`/propsync` performs a full rebuild and unified-memory synchronization.

Normal runtime retrieval continues to use the stable subject composition path,
but the gate diagnostics now expose predicate types when typed metadata exists.

## Verification

~~~powershell
python .\run_typed_subject_proposition_v10100.py
python .\run_typed_semantic_end_to_end_v10100.py
~~~

Expected final statuses:

~~~text
TYPED_SUBJECT_PROPOSITION_PASS
TYPED_SEMANTIC_END_TO_END_PASS
~~~

This release establishes the first typed semantic layer for the next stage of
relation-aware composition, decomposition, and semantic routing.
