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


## Typed Predicate Query Routing

v10.10.0 now uses predicate types not only as metadata but also as a runtime
retrieval key.

The new query path is:

~~~text
Question
  ↓
Predicate Type Detection
  ↓
Typed Proposition Filtering
  ↓
Answer Composition
~~~

Examples:

~~~text
GPUの性質は
    -> property
    -> GPUは、高速であり、低消費電力である。

GPUは何が得意
    -> capability
    -> GPUは、並列計算が得意である。

GPUとCUDAの関係は
    -> relation
    -> GPUは、CUDAを利用可能である。
~~~

Typed routing is conservative and requires an explicit cue. Generic questions
continue to use the existing full subject composition path, preventing the new
feature from narrowing answers unexpectedly.

### Query cues

~~~text
property   : 性質 / 特徴
capability : 得意 / できる / 能力 / 機能
relation   : 関係 / 利用 / 依存 / 接続
definition : とは / 定義
~~~

### Verification

~~~powershell
python .\run_typed_query_routing_v10100.py
python .\run_typed_query_end_to_end_v10100.py
~~~

Expected final statuses:

~~~text
TYPED_QUERY_ROUTING_PASS
TYPED_QUERY_END_TO_END_PASS
~~~


## Internalized Knowledge Route

v10.10.0 now distinguishes knowledge that has actually been consumed by
incremental training from external semantic retrieval.

A concept is marked INTERNALIZED only when:

~~~text
trusted teaching pair
      ↓
/train
      ↓
pair fingerprint recorded in chat_learning_state.json
      ↓
Internalized Knowledge Registry
~~~

Merely adding a teaching pair is not sufficient.

### Runtime order

~~~text
Typed Semantic Retrieval
      ↓ miss
Canonical Definition Retrieval
      ↓ miss
Unified Semantic Memory
      ↓ miss
Internalized Knowledge Route
      ├ INTERNALIZED -> model generation + normal output gate
      └ otherwise    -> Unknown Gate
~~~

The internalized registry stores provenance only. It does not return the
teacher answer directly. Runtime answers are generated from the trained model
weights and must still pass confidence, agreement and semantic consistency
checks.

When accepted, diagnostics report:

~~~text
gate=INTERNALIZED
route=internalized model generation
~~~

If the model output fails the normal gate, the response remains UNKNOWN rather
than being force-accepted.

### Inspection

~~~text
/internalized
~~~

lists concepts whose trusted teaching pairs are proven to have been consumed
by /train.

### Verification

~~~powershell
python .\run_internalized_registry_v10100.py
python .\run_internalized_route_v10100.py
~~~

Expected final statuses:

~~~text
INTERNALIZED_REGISTRY_PASS
INTERNALIZED_ROUTE_PASS
~~~


## Internalized Registry Consolidation

The fingerprint-level internalized registry is now exposed through a
concept-level consolidated view.

Before:

~~~text
宇宙
文学
架空装置
架空装置
文学
文学
文学
量子センサー
~~~

After consolidation:

~~~text
宇宙          trained_pairs=1
文学          trained_pairs=4
架空装置      trained_pairs=2
量子センサー  trained_pairs=1
~~~

Each consolidated concept retains:

~~~text
concept
latest_question
latest_teacher_answer
latest_source
fingerprints[]
sources[]
trained_pairs
~~~

Fingerprint-level evidence is preserved for auditability. Consolidation affects
the registry view only and does not change the runtime proof requirement for an
INTERNALIZED route.

The chat command:

~~~text
/internalized
~~~

now reports one row per concept.

### Verification

~~~powershell
python .\run_internalized_consolidation_v10100.py
~~~

Expected final status:

~~~text
INTERNALIZED_CONSOLIDATION_PASS
~~~
