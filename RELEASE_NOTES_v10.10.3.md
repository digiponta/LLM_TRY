# LLM_TRY v10.10.3 — Truth State

v10.10.3 adds an explicit truth-state overlay on top of Knowledge State,
Dispatcher, and Provenance.

## Truth states

~~~text
TRUE
FALSE
UNVERIFIED
CONTESTED
OUTDATED
~~~

Truth records are stored separately from model weights and semantic memory:

~~~text
data/truth_state_v10103.jsonl
~~~

Knowledge is retained even when marked FALSE, CONTESTED, or OUTDATED. The
truth overlay controls runtime use without silently deleting the original
knowledge.

## Runtime policy

~~~text
TRUE
  -> keep existing dispatcher action

UNVERIFIED
  -> keep existing action + warning

CONTESTED
  -> keep existing action + warning

FALSE
  -> correction present: RETRIEVE correction + warning
  -> no correction: BLOCK

OUTDATED
  -> correction present: RETRIEVE correction + warning
  -> no correction: BLOCK
~~~

The policy also applies to INTERNALIZED knowledge. A FALSE internalized concept
with a stored correction is not generated from model weights; the correction
is returned instead.

## Default state

Concepts without an explicit truth record are conservatively treated as:

~~~text
UNVERIFIED
~~~

This does not by itself change the existing Dispatcher action. It adds a
warning so legacy knowledge remains usable while truth metadata is gradually
curated.

## Commands

~~~text
/truth CONCEPT
/truths
/truthset CONCEPT TRUE
/truthset CONCEPT CONTESTED
/truthset CONCEPT FALSE => CORRECTION
/truthset CONCEPT OUTDATED => CORRECTION
~~~

The /dispatch command is truth-aware in v10.10.3.

## Architecture

~~~text
User Query
   ↓
Knowledge State Resolver
   ↓
Knowledge State Dispatcher
   ↓
Truth State Overlay
   ├─ TRUE       -> preserve
   ├─ UNVERIFIED -> warn
   ├─ CONTESTED  -> warn
   ├─ FALSE      -> correct or block
   └─ OUTDATED   -> correct or block
   ↓
Retrieval / Generation / Block
~~~

## Verification

~~~powershell
python .\run_truth_state_store_v10103.py
python .\run_truth_aware_dispatch_v10103.py
python .\run_truth_state_end_to_end_v10103.py
~~~

Expected statuses:

~~~text
TRUTH_STATE_STORE_PASS
TRUTH_AWARE_DISPATCH_PASS
TRUTH_STATE_END_TO_END_PASS
~~~
