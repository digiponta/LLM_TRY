# LLM_TRY v10.11.0 — Semantic Knowledge Architecture

v10.11.0 integrates the semantic knowledge experiments developed across
v10.9.x and v10.10.x into one runtime/control architecture.

## Architecture

~~~text
Natural-language query
        ↓
Semantic Knowledge Architecture
        │
        ├─ Atomic Proposition Store
        ├─ Subject-Keyed Index
        ├─ Typed Predicate Index
        ├─ Unified Semantic Memory
        ├─ Internalized Knowledge Registry
        ├─ Provenance / Source Tracking
        ├─ Truth State Overlay
        ├─ Knowledge State Resolver
        └─ Knowledge State Dispatcher
        │
        ↓
RETRIEVE / GENERATE / BLOCK
        ↓
Normal model output gate when generation is selected
~~~

## Single runtime API

Runtime clients can now use:

~~~python
result = semantic_knowledge.resolve(query)
~~~

The result contains:

~~~text
knowledge_state
base_dispatch
dispatch
truth
truth_result
provenance
action
answer
focus
~~~

This replaces duplicated resolver/dispatcher/truth orchestration in chat.py.

## SemanticKnowledgeConfig

The architecture receives all semantic stores through one configuration:

~~~text
proposition_path
subject_index_path
typed_index_path
unified_path
learning_log
learning_state
raw_corpus_path
truth_store_path
canonical_definitions
~~~

This makes the semantic layer portable to future QHA integration without
hard-wiring chat.py paths into the architecture.

## Knowledge write path

/propteach-style semantic teaching is represented internally as:

~~~text
Natural-language proposition
        ↓
Atomic Proposition
        ↓
Subject Index rebuild
        ↓
Typed Index rebuild
        ↓
Unified Semantic Memory upsert
~~~

In chat.py, /propteach now calls the architecture facade instead of manually
coordinating each layer.

/propsync similarly calls one full synchronization operation.

## Runtime read path

~~~text
query
  ↓
resolve knowledge state
  ↓
attach provenance
  ↓
dispatch
  ↓
apply truth overlay
  ↓
SemanticKnowledgeResult
~~~

The existing rules remain unchanged:

~~~text
TYPED           -> RETRIEVE
CANONICAL       -> RETRIEVE
UNIFIED         -> RETRIEVE
INTERNALIZED    -> GENERATE
RAW_CORPUS_ONLY -> BLOCK
UNKNOWN         -> BLOCK
NON_CONCEPT     -> GENERATE
~~~

Truth State may restrict or correct the result, but does not promote weak
knowledge. For example, RAW_CORPUS_ONLY remains BLOCK even if marked TRUE.

## Commands

Existing commands remain compatible.

New architecture inspection:

~~~text
/semstatus
~~~

The diagnostic commands:

~~~text
/kstate
/dispatch
/provenance
~~~

now resolve through the same Semantic Knowledge Architecture used by the normal
runtime, eliminating diagnostic/runtime policy drift.

## Verification

~~~powershell
python .\run_semantic_knowledge_architecture_v10110.py
python .\run_semantic_knowledge_compatibility_v10110.py
~~~

Expected statuses:

~~~text
SEMANTIC_KNOWLEDGE_ARCHITECTURE_PASS
SEMANTIC_KNOWLEDGE_COMPATIBILITY_PASS
~~~

Previous v10.10.x regressions should also remain PASS.

## Position toward QHA

v10.11.0 provides a clean semantic subsystem boundary suitable for QHA:

~~~text
QHA / Semantic Task Layer
        ↓
SemanticKnowledgeArchitecture.resolve()
        ↓
semantic state + truth + provenance + action
        ↓
QHA scheduler / executor selection
~~~

This makes Semantic Knowledge a subsystem rather than a collection of
chat-specific routing features.
