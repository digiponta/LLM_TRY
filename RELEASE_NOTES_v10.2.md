# LLM_TRY v10.2 — Unknown-to-Teaching Learning Loop

## Goal

v10.2 extends the stable v10.1 gate baseline with a safe control-plane path for
turning an unknown concept into an explicit teaching candidate.

The language model is still not allowed to learn automatically from its own
rejected answer. Human teaching remains required.

## Flow

```text
User question
   |
   v
Pre-generation unknown concept gate
   |
   +-- KNOWN ----------------------------> normal LLM path
   |
   +-- UNKNOWN
          |
          v
      "未学習です"
          |
          v
 data/knowledge_queue.jsonl
          |
          v
 /teach ANSWER
   or
 /teachq QUESTION => ANSWER
          |
          v
 trusted learning pair
          |
          v
 /train
          |
          v
 incremental checkpoint
```

## Changes

- Pre-generation UNKNOWN decisions are now persisted to the knowledge queue.
- Repeated UNKNOWN requests are deduplicated by the existing route fingerprint.
- The interactive chat prints the teaching path immediately after an UNKNOWN.
- Successful `/teach` and `/teachq` operations resolve matching
  `UNKNOWN_KNOWLEDGE` queue entries.
- Added a generic `resolve_route_queue()` helper.
- Added `eval_unknown_teaching_loop_v102.py`.

## Safety property

v10.2 does not train on an untrusted model candidate. The sequence is:

```text
UNKNOWN detection
-> queue
-> explicit human teaching
-> teaching validation
-> trusted learning log
-> explicit /train
```

This preserves the v10.x design principle that gate decisions and language
generation remain separated.

## Verification

```powershell
python eval_unknown_teaching_loop_v102.py
python eval_known_false_rejection_v96.py
python eval_integrated_gate_v97.py
python eval_multiturn_history_v101.py
```

The new v10.2 regression should report all PASS before the branch is merged
into `main`.
