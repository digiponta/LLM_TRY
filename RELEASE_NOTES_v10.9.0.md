# LLM_TRY v10.9.0 — Semantic Integration

This branch begins integration of validated LLM_SEM semantic mechanisms into
LLM_TRY without replacing the existing adaptive-learning or Known/Unknown
control plane.

## Phase 1: Semantic Proposition Compose / Decompose

The first integrated component is a bidirectional atomic proposition layer.

~~~text
XはYである。
XはZである。
    ↓ decompose
(X, Y)
(X, Z)
    ↓ compose
Xは、Yであり、Zである。
~~~

The reverse direction is also supported:

~~~text
Xは、Yであり、Zである。
    ↓
(X, Y)
(X, Z)
~~~

### Files

- `semantic_proposition_v1090.py`
- `run_semantic_proposition_integration_v1090.py`
- `run_semantic_proposition_chat_regression_v1090.py`

### Chat integration

`chat.py` adds:

~~~text
/propteach <statement>
/prop <subject>
/props
~~~

Normal concept queries now check the atomic proposition store before the
existing unified semantic-memory lookup and before model generation.

Example:

~~~text
/propteach 架空装置は高速である。
/propteach 架空装置は低消費電力である。
架空装置とは
~~~

Expected answer:

~~~text
架空装置は、高速であり、低消費電力である。
~~~

This path is retrieval/composition, not model generation.

## Design constraint

The existing LLM_TRY mechanisms remain unchanged in this phase:

- adaptive training
- Known/Unknown gate
- teaching queues
- unified semantic memory
- relation/fact behavior
- online checkpoint selection

The proposition layer is intentionally isolated so it can be regression-tested
before deeper integration with semantic routing and model internalization.

## Verification

~~~powershell
python .\run_semantic_proposition_integration_v1090.py
python .\run_semantic_proposition_chat_regression_v1090.py
python .\chat.py
~~~

Phase 2 will connect proposition normalization with the existing unified
semantic-memory/relation pipeline. Later phases can add LLM_SEM-style semantic
routing and controlled internalization.


## Phase 2: Atomic Proposition Store + Unified Semantic Memory

Phase 2 connects the atomic proposition store to the existing unified semantic
memory. A successful `/propteach` now:

~~~text
Natural-language proposition
        ↓
decompose
        ↓
Atomic Proposition Store
        ↓
compose same subject
        ↓
Unified Semantic Memory upsert
        ↓
normal semantic retrieval
~~~

Example:

~~~text
/propteach 架空装置は高速である。
/propteach 架空装置は低消費電力である。
~~~

The unified semantic-memory row becomes:

~~~json
{
  "concept": "架空装置",
  "assistant": "架空装置は、高速であり、低消費電力である。",
  "source": "atomic-proposition",
  "atomic_count": 2
}
~~~

Existing unrelated unified-memory rows are preserved. Existing rows for the
same concept are replaced by one canonical composed row.

The chat command `/propsync` performs a full synchronization of every subject
currently stored in the atomic proposition database.

### Phase 2 verification

~~~powershell
python .\run_phase2_atomic_unified_regression_v1090.py
python .\chat.py
~~~

Interactive check:

~~~text
/propteach 架空装置は高速である。
/propteach 架空装置は低消費電力である。
/props
架空装置とは
~~~

Expected result:

~~~text
架空装置は、高速であり、低消費電力である。
~~~

The resulting unified-memory record remains compatible with the existing
`semantic_knowledge_lookup()` path.


## Subject-Keyed Semantic Proposition Index

v10.9.0 now adds a derived subject-keyed semantic index on top of the atomic
proposition store.

Conceptually:

~~~text
GPUは高速である。
    ↓
GPU => GPUは高速である。
~~~

Multiple predicates are indexed independently:

~~~text
GPU => GPUは高速である。
GPU => GPUは並列計算が得意である。
~~~

and can be recomposed into:

~~~text
GPUは、高速であり、並列計算が得意である。
~~~

### Source of truth

The atomic proposition store remains authoritative.

~~~text
Atomic Proposition Store
        ↓ rebuild
Subject-Keyed Proposition Index
        ↓ compose
Unified Semantic Memory
        ↓
Runtime Retrieval
~~~

The derived index is stored at:

~~~text
data/subject_keyed_propositions_v1090.jsonl
~~~

Each row uses:

~~~json
{
  "subject": "GPU",
  "statement": "GPUは高速である。",
  "value": "高速"
}
~~~

### Chat commands

~~~text
/subject GPU
/subjects
~~~

`/propteach` automatically rebuilds the subject index before updating unified
semantic memory. `/propsync` rebuilds both the subject index and unified
semantic memory.

Normal concept queries consult the subject-keyed index before the lower-level
atomic proposition lookup.

### Verification

~~~powershell
python .\run_subject_keyed_proposition_regression_v1090.py
python .\run_subject_keyed_end_to_end_v1090.py
~~~

Expected final statuses:

~~~text
SUBJECT_KEYED_PROPOSITION_PASS
SUBJECT_KEYED_END_TO_END_PASS
~~~
