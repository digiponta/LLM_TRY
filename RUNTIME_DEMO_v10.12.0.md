# v10.12.0 Multi-Item Batch Repair Runtime Demonstration

This procedure demonstrates the real multi-item batch-repair runtime path.

The preparation step is deterministic so the test does not depend on waiting for two naturally unstable generations.

## 1. Prepare two active repair tasks

Exit chat.py first.

Run:

```powershell
python .\prepare_batch_repair_runtime_v10120.py --dry-run
```

Default target concepts:

- 文学
- 架空装置

The dry run must report at least two CURRENT internalized concepts.

Then stage them:

```powershell
python .\prepare_batch_repair_runtime_v10120.py
```

The script:

- backs up data/chat_learning_state.json
- creates two verification repair tasks
- removes only the selected trusted fingerprints from learning-state
- marks both tasks retrain
- does not modify checkpoint weights

## 2. Start chat with the current checkpoint

Use the checkpoint that currently contains the trusted fingerprints.

Example:

```powershell
python .\chat.py `
  --model .\model\model-gpu-v1.6.2-online.pt `
  --tokenizer .\model\tokenizer-v0.7-bpe.json
```

## 3. Inspect batch

Inside chat:

```text
/batchstatus
/verifications
```

Expected shape:

```text
[verification batch: tasks=2, fingerprints=2, concepts=2]
  01. concept='文学' status=retrain ...
  02. concept='架空装置' status=retrain ...
```

## 4. Batch repair

Run:

```text
/trainbatch
```

Expected training characteristics:

```text
New/rebind      : 2
Prior trusted   : remaining trusted pairs
Consumed        : 2 new trusted pair(s)
```

Only one online_train.py invocation should occur.

## 5. Re-verify each concept

Query:

```text
文学
架空装置
```

Each INTERNALIZED concept must pass the normal generation gate and Composite Teacher Fidelity Gate before its verification task becomes verified.

## 6. Confirm lifecycle

Run:

```text
/verifystatus
/verifications
```

Expected after both pass:

```text
pending=0
retrain=0
verified=<previous verified + 2>
failed=0
```

and no active tasks for these two concepts.

## What is synthetic vs real

Synthetic:

- creation of the initial repair tasks only

Real:

- trusted-pair reactivation
- one batch online training run
- checkpoint save/reload
- checkpoint binding refresh
- model generation after training
- Composite Teacher Fidelity evaluation
- per-concept transition to verified

This separates task creation reproducibility from actual repair-learning behavior.
