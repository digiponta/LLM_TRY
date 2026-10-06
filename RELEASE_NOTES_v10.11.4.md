# LLM_TRY v10.11.4 — Knowledge Promotion Pipeline

v10.11.4 connects UNKNOWN_KNOWLEDGE intake to validated Semantic Knowledge.

## Pipeline

Bare unknown concept
  -> UNKNOWN / BLOCK
  -> knowledge_queue.jsonl
  -> explicit promotion teaching
  -> structural proposition validation
  -> Atomic Proposition
  -> Subject Index
  -> Typed Index
  -> Unified Semantic Memory
  -> queue status = promoted
  -> bare semantic routing enabled

## Commands

/promotions

/promote CONCEPT => STATEMENT

Example:
/promote 時間 => 時間は、出来事の順序と間隔を表す概念である。

After promotion:
時間 -> semantic bare routing -> 時間とは -> TYPED -> RETRIEVE

## Safety conditions

Promotion requires:
1. a pending UNKNOWN_KNOWLEDGE queue entry for the concept;
2. a valid semantic proposition statement;
3. every parsed proposition subject must match the promoted concept.

A mismatched statement such as `/promote 時間 => 宇宙は、広いものである。` is rejected.

## Truth State

Promotion does NOT automatically set Truth State to TRUE.
A promoted concept defaults to UNVERIFIED until explicitly reviewed with /truthset.
This separates knowledge availability from truth verification.

## Queue state

Successfully promoted UNKNOWN_KNOWLEDGE entries are annotated with:
status=promoted
promoted_at
promoted_concept
promoted_statement
post_knowledge_state

They are no longer returned by /promotions.

## Regression

python .\run_knowledge_promotion_pipeline_v10114.py

Expected:
STATUS : KNOWLEDGE_PROMOTION_PIPELINE_PASS

Existing v10.11.x regressions should remain PASS.
