# LLM_TRY v10.10.1 — Knowledge State Dispatcher

v10.10.1 promotes the v10.10.0 Knowledge State Resolver from a diagnostic layer
to the single runtime routing authority.

## Runtime architecture

~~~text
User Query
   ↓
Knowledge State Resolver
   ↓
Knowledge State Dispatcher
   ├─ TYPED           -> RETRIEVE
   ├─ CANONICAL       -> RETRIEVE
   ├─ UNIFIED         -> RETRIEVE
   ├─ INTERNALIZED    -> GENERATE + normal output gate
   ├─ RAW_CORPUS_ONLY -> BLOCK
   ├─ UNKNOWN         -> BLOCK
   └─ NON_CONCEPT     -> GENERATE
~~~

The previous independent runtime branches for typed lookup, canonical lookup,
unified lookup, internalized detection and concept pre-gating have been removed
from the main answer path. Their evidence is now resolved once, then dispatched
once.

## Dispatcher actions

~~~text
RETRIEVE
  deterministic answer; zero model-generation tokens

GENERATE
  model generation allowed; normal confidence/agreement/semantic gate remains

BLOCK
  answer generation prohibited; route to UNKNOWN_KNOWLEDGE / teaching
~~~

This ensures that RAW_CORPUS_ONLY cannot accidentally fall through to model
generation.

## Compatibility

Bare validated concepts such as CPU continue to use the existing canonical
query normalization before resolution.

INTERNALIZED answers continue to use the trained model and must pass the normal
output gate. They are never force-accepted.

## Inspection

~~~text
/kstate QUERY
/dispatch QUERY
~~~

/kstate shows the Resolver result. /dispatch shows the actual runtime action.

## Verification

~~~powershell
python .\run_knowledge_state_dispatcher_v10101.py
python .\run_knowledge_dispatch_end_to_end_v10101.py
~~~

Expected statuses:

~~~text
KNOWLEDGE_STATE_DISPATCHER_PASS
KNOWLEDGE_DISPATCH_END_TO_END_PASS
~~~
