# LLM_TRY v10.11.5 — Knowledge Queue Lifecycle Consolidation

v10.11.5 gives UNKNOWN_KNOWLEDGE one canonical lifecycle.

## Canonical lifecycle

pending -> promoted -> verified

- pending: unknown knowledge request exists but no validated semantic knowledge is available.
- promoted: the concept has been promoted into validated Semantic Knowledge.
- verified: the promoted concept has explicit Truth State TRUE.

Truth State FALSE, UNVERIFIED, CONTESTED, or OUTDATED revokes verified back to promoted.

## Removed duplicate lifecycle writers

The semantic knowledge queue is no longer changed to legacy `resolved` by:

- normal semantic RETRIEVE
- /teach
- /teachq

Those operations may affect answers or learning data, but they do not own the knowledge queue lifecycle.

Only the lifecycle manager owns semantic knowledge queue state.

## Promotion

/promote X => XはYである。

transitions matching pending rows to promoted.

The v10.11.4 `queue_resolved` result name remains as a compatibility property, while v10.11.5 uses `queue_promoted` as the canonical term.

## Verification

/truthset X TRUE

transitions matching promoted rows to verified.

If Truth State later changes away from TRUE, verified rows return to promoted and record verification_revoked_at.

## Legacy consolidation

Older versions may have written status=resolved after a successful retrieval.

At chat startup, v10.11.5 safely converts a legacy resolved row only when another row for the same concept already proves promoted or verified lifecycle state.

This handles the v10.11.4 case where promotion was followed by a second retrieval-time resolved update.

## Commands

/promotions
  Lists pending UNKNOWN_KNOWLEDGE requests.

/promotionstatus
  Shows counts for pending, promoted, verified, legacy_resolved, other, and total.

## Regression

python .\run_knowledge_queue_lifecycle_v10115.py

Expected:

STATUS : KNOWLEDGE_QUEUE_LIFECYCLE_PASS

Also rerun:

python .\run_knowledge_promotion_pipeline_v10114.py
python .\run_bare_unknown_safety_gate_v10113.py
python .\run_bare_concept_semantic_routing_v10112.py

Existing v10.11.x regressions should remain PASS.
