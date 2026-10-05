# LLM_TRY v10.11.6 — Internalized Checkpoint Binding

v10.11.6 binds INTERNALIZED knowledge to the checkpoint that actually contains its trained fingerprints.

## Problem

Before v10.11.6, INTERNALIZED was derived only from chat_history.jsonl + chat_learning_state.json.
After switching to a separately trained batch checkpoint, the registry could claim a concept was internalized even when the current model weights had never consumed that pair.

Observed example:

量子センサー
  -> registry says INTERNALIZED
  -> batch checkpoint lacks the pair
  -> bad model candidate
  -> output gate rejects

## New states

INTERNALIZED
  Registry evidence exists AND current checkpoint contains at least one matching trained fingerprint.

INTERNALIZED_STALE
  Registry evidence exists BUT current checkpoint contains no matching fingerprint.

INTERNALIZED_STALE is BLOCK and never reaches model generation.

## Checkpoint metadata

LanguageModel.save_checkpoint now accepts metadata.
online_train.py writes:

knowledge_binding_version=v10.11.6
trained_fingerprints=[...]
base_model=...
chat_data=...

## Rebinding

online_train.py now compares trusted pairs with both:

- chat_learning_state.json
- current checkpoint trained_fingerprints

A pair is train/rebind pending when it is absent from either required training state.
This preserves explicit reactivation while also allowing a new batch checkpoint to replay trusted knowledge.

## Runtime

chat.py passes checkpoint fingerprints into SemanticKnowledgeArchitecture.
After /train reload, the architecture is rebuilt with the new checkpoint binding.

/internalized now reports binding=CURRENT or binding=STALE.

STALE runtime:

INTERNALIZED_STALE
  -> BLOCK
  -> checkpoint rebind required
  -> /train

The stale path does not enqueue the concept as new UNKNOWN knowledge.

## Regression

python .\run_internalized_checkpoint_binding_v10116.py

Expected:

STATUS : INTERNALIZED_CHECKPOINT_BINDING_PASS

## Batch checkpoint migration

A v10.11.5 batch checkpoint has no binding metadata by design, so registry-only internalized concepts appear STALE.
Start chat with a distinct --online-output, then run /train to replay trusted pairs into the batch checkpoint lineage.
