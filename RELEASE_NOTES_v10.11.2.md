# LLM_TRY v10.11.2 — Bare Concept Semantic Routing

v10.11.2 lets validated semantic concepts enter the Semantic Knowledge
Architecture even when the user supplies only the concept name.

## Goal

Before v10.11.2:

~~~text
宇宙
  -> normal model generation

宇宙とは
  -> UNIFIED semantic retrieval
~~~

After v10.11.2:

~~~text
宇宙
  -> semantic bare routing
  -> 宇宙とは
  -> UNIFIED semantic retrieval
~~~

The routing is intentionally selective.

## Validated semantic sources

A bare concept is canonicalized only when the concept already exists in one of
these validated layers:

~~~text
Subject / Typed Proposition
Canonical Definitions
Unified Semantic Memory
Internalized Knowledge
~~~

Raw corpus frequency alone is not sufficient.

## Preservation rules

~~~text
GPU
  -> TYPED

CPU
  -> CANONICAL
  -> Truth State still applies

宇宙
  -> UNIFIED

量子センサー
  -> INTERNALIZED

長門
  -> normal generation unless explicitly present in Semantic Knowledge

時間
  -> raw corpus occurrence does not promote bare semantic routing
~~~

This preserves persona behavior while allowing semantic knowledge to answer
bare concept queries.

## Architecture API

SemanticKnowledgeResult now records:

~~~text
query
routed_query
bare_concept_routed
knowledge_state
dispatch
truth
provenance
~~~

Example:

~~~text
query='宇宙'
routed_query='宇宙とは'
bare_concept_routed=True
state=UNIFIED
action=RETRIEVE
~~~

## Runtime

chat.py now passes the original user input directly to:

~~~python
semantic_knowledge.resolve(user_text)
~~~

The old fixed-list runtime canonicalization is no longer used for routing.

When a bare concept is routed, chat.py prints:

~~~text
[semantic bare routing: '宇宙' -> '宇宙とは']
~~~

## Truth State

Truth policy is applied after bare routing.

For example:

~~~text
CPU
  -> CPUとは
  -> CANONICAL
  -> Truth State = FALSE
  -> BLOCK
~~~

Bare routing never bypasses Provenance or Truth State.

## Regression

~~~powershell
python .\run_bare_concept_semantic_routing_v10112.py
~~~

Expected:

~~~text
STATUS : BARE_CONCEPT_SEMANTIC_ROUTING_PASS
~~~

Existing v10.11.x regressions should remain PASS.
