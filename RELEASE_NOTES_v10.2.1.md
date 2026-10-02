# LLM_TRY v10.2.1 — Trained Concept Promotion

## Purpose

v10.2.1 closes the remaining gap in the v10.2 learning loop.

In v10.2, an unknown concept could be queued, explicitly taught, and consumed by
incremental training. However, the pre-generation concept gate still used a
static known-concept list. That meant a newly trained concept could still be
rejected before generation.

v10.2.1 promotes a concept into the pre-generation KNOWN set only when the
exact trusted teaching pair has actually been consumed by `/train`.

## Promotion rule

```text
UNKNOWN
  -> /teach
  -> trusted learning log
  -> still UNKNOWN

UNKNOWN
  -> /teach
  -> /train
  -> fingerprint recorded in chat_learning_state.json
  -> concept promoted
  -> pre-generation gate allows generation
```

Teaching alone is therefore insufficient. Promotion requires evidence that the
pair was consumed by incremental training.

## Added

- `trained_known_concepts()`
- optional `promoted_concepts` input to `pre_generation_unknown_concept()`
- runtime lookup of trained concepts before the pre-generation unknown gate
- `eval_trained_concept_promotion_v1021.py`

## Verification

```powershell
python eval_trained_concept_promotion_v1021.py
python eval_unknown_teaching_loop_v102.py
python eval_known_false_rejection_v96.py
python eval_integrated_gate_v97.py
python eval_multiturn_history_v101.py
```

## Real-model experiment

After the control-plane regressions pass:

```text
python chat.py

You> 宇宙とは
AI> 未学習です

You> /teach 宇宙は、物質・エネルギー・時空を含む世界全体を指す。
You> /train
You> 宇宙とは
```

The final response should now reach the trained language model instead of being
blocked by the static unknown-concept gate.
