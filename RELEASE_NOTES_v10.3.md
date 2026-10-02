# LLM_TRY v10.3 — Adaptive Learning Preservation Regression

## Purpose

v10.2.4 demonstrated a complete persistent adaptive-learning loop:

```text
UNKNOWN
-> explicit /teach
-> single-pair GPU /train
-> trained concept promotion
-> saved online checkpoint
-> process restart
-> automatic online checkpoint resume
-> learned concept remains KNOWN
```

v10.3 verifies the next requirement: the new learned concept must not damage
the stable baseline or accidentally promote unrelated unknown concepts.

## New regression

`eval_adaptive_preservation_v103.py` checks three properties:

1. Stable baseline preservation
   - Nagato identity/persona
   - AI
   - LLM
   - CUDA
   - quantum mechanics

2. Learned concept persistence
   - `宇宙とは`
   - exact taught answer
   - gate remains KNOWN / ACCEPT

3. Unknown isolation
   - mathematics
   - literature
   - black holes
   - relativity
   remain rejected unless explicitly taught and trained.

## Run

```powershell
python eval_adaptive_preservation_v103.py
```

Expected result:

```text
Passed            : 16/16
Failed            : 0/16
Regression status : PASS
```

If this passes, the experiment demonstrates persistent concept-specific
incremental learning while preserving the established v10.x behavior.
