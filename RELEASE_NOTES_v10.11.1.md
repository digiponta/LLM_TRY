# LLM_TRY v10.11.1 — Semantic Knowledge Snapshot / Inspection

v10.11.1 adds a read-only concept inspection layer on top of the
v10.11.0 Semantic Knowledge Architecture.

## Goal

A single concept can now be inspected across all semantic knowledge layers:

~~~text
Concept
  ├─ Atomic Proposition
  ├─ Subject Index
  ├─ Typed Predicate Index
  ├─ Unified Semantic Memory
  ├─ Internalized Knowledge
  ├─ Provenance
  ├─ Truth State
  ├─ Knowledge State
  └─ Final Dispatch
~~~

The snapshot does not mutate any semantic store.

## Snapshot API

~~~python
snapshot = snapshot_semantic_knowledge(
    semantic_knowledge,
    "GPU",
)
~~~

The result exposes:

~~~text
concept
query
atomic_propositions
subject_answer
typed_rows
unified_row
internalized
result
~~~

Convenience fields include:

~~~text
proposition_count
typed_count
unified_present
internalized_present
truth_state
final_action
~~~

## Chat inspection command

~~~text
/semantic CONCEPT
~~~

Examples:

~~~text
/semantic GPU
/semantic CPU
/semantic 宇宙
/semantic 量子センサー
/semantic 時間
~~~

The command prints one cross-layer snapshot without changing the stored
knowledge.

## Example interpretation

For an internalized concept with a FALSE truth correction:

~~~text
Internalized : yes
Provenance   : trained fingerprint evidence
Truth State  : FALSE
Final Route  : RETRIEVE / truth-state correction
~~~

This makes it possible to see that model weights still contain the learned
concept while runtime policy prevents use of the false knowledge.

For RAW_CORPUS_ONLY:

~~~text
Proposition  : none
Typed Index  : none
Unified      : none
Internalized : no
Knowledge    : RAW_CORPUS_ONLY
Final Route  : BLOCK
~~~

This shows why corpus occurrence does not become trusted semantic knowledge.

## Files

~~~text
semantic_knowledge_snapshot_v10111.py
run_semantic_knowledge_snapshot_v10111.py
~~~

## Verification

~~~powershell
python .\run_semantic_knowledge_snapshot_v10111.py
~~~

Expected status:

~~~text
SEMANTIC_KNOWLEDGE_SNAPSHOT_PASS
~~~

The v10.11.0 architecture and compatibility regressions should remain PASS:

~~~powershell
python .\run_semantic_knowledge_architecture_v10110.py
python .\run_semantic_knowledge_compatibility_v10110.py
~~~

## QHA relevance

The snapshot acts as a semantic observability interface. A future QHA runtime
can inspect why a Semantic Task was routed to RETRIEVE, GENERATE, or BLOCK
without directly reading each underlying store.
