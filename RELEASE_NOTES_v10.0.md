# LLM_TRY v10.0 Release Notes

## Summary

v10.0 freezes the successful v9.x experimental line as a reproducible
baseline for a small Nagato-style Japanese assistant.

The main architectural result is the separation of:

- **LM generation for known concepts**
- **pre-generation unknown-concept routing**
- **post-generation semantic/category/confidence validation**

## Model

```text
Parameters      : 8,960,000
Vocabulary      : 8,000
d_model         : 256
Layers          : 6
Heads           : 8
Context         : 512
Base checkpoint : model/model-gpu-v0.8-chat-clean.pt
SFT checkpoint  : model/model-llm-try-nagato-chat-v94.pt
```

## Canonical Weighted SFT

Training uses `data/nagato_canonical_v91.jsonl` with one canonical answer
per normalized prompt.

Weights:

```text
identity   x12
persona    x6
knowledge  x5
paraphrase x3
general    x1
```

Fine-tuning remains partial:

```text
Blocks 5-6 + FinalNorm : 5e-6
LM Head                : 1e-6
```

## Unknown Handling

v9.3 showed that training many `未学習です` targets directly into the LM can
spill into known prompts. v9.4-v9.6 moved this responsibility outside the LM.

Final flow:

```text
User
  -> Pre-generation Concept Gate
       -> Unknown : 未学習です
       -> Known   : LLM
                     -> Semantic / Category / Confidence Gate
                     -> Answer
```

## Final Regression

```powershell
python eval_integrated_gate_v97.py
```

Verified result:

```text
Known preservation : 11/11 = 100.0%
Unknown rejection  : 20/20 = 100.0%
Balanced accuracy  : 100.0%
Parse failures     : 0
Regression status  : PASS
```

This result applies to the fixed 31-prompt regression set and should not be
interpreted as general-purpose model accuracy.

## Notable Fixes

- canonical dataset conflict resolution
- canonical weighted SFT
- independent checkpoint naming
- Windows UTF-8 subprocess handling in evaluation
- pre-generation concept gate
- known false-rejection diagnostic
- category-aware definition evidence
- integrated Known/Unknown regression test

## Key Files

```text
train_nagato_chat.py
build_nagato_canonical_dataset.py
audit_nagato_training_conflicts.py
eval_unknown_gate_v94.py
eval_known_false_rejection_v96.py
eval_integrated_gate_v97.py
data/nagato_canonical_v91.jsonl
data/nagato_unknown_paraphrase.jsonl
```

## Status

v10.0 is an experimental stable baseline suitable for further work on:

- broader held-out concept tests
- automatic known-concept registration
- semantic rather than lexical pre-generation routing
- incremental teaching without catastrophic interference
