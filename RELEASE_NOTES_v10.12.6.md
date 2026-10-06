# LLM_TRY v10.12.6 — Nagato Knowledge Gain Evaluation

v10.12.6 measures whether moderate data-nagato.txt continued pretraining actually adds model knowledge while preserving verified behavior.

## Evaluation dimensions

1. Held-out raw-corpus NLL / perplexity
2. Per-window improvement rate
3. Optional QA probe gain
4. Post-Nagato INTERNALIZED retention
5. Persona retention

The raw-corpus metrics are the primary learning-gain signal because data-nagato.txt is raw text rather than a QA dataset.

## Main evaluator

Run:

python .\evaluate_nagato_knowledge_gain_v10126.py \
  --before model\model-gpu-v1.6.2-online-pre-nagato.pt \
  --after model\model-gpu-v1.6.2-online.pt

The report is written to:

results/nagato_knowledge_gain_v10126.json

Important outputs include:

- Before mean NLL
- After mean NLL
- NLL gain
- Relative gain
- Before / After perplexity
- Improved / unchanged / regressed held-out windows
- INTERNALIZED retention
- Persona retention

## Optional QA probes

If present, the evaluator reads:

data/nagato_gain_probes_v10126.jsonl

Each row should contain:

{"concept":"...", "question":"...", "answer":"..."}

Only source-supported probes should be added. No probes are invented automatically from the corpus.

## Baseline snapshot

train_nagato_moderate_v10124.py now preserves the pre-training checkpoint at:

model/model-gpu-v1.6.2-online-pre-nagato.pt

The snapshot is created only when missing and is never overwritten automatically.

This makes future before/after gain measurements reproducible.

## Existing promoted model

The first v10.12.5 promotion happened before permanent snapshot support was added. Therefore, the exact immediately-pre-Nagato checkpoint may not exist locally.

For the current promoted model:

- if an older pre-Nagato checkpoint was manually retained, pass it with --before
- otherwise, the exact first-pass before/after comparison cannot be reconstructed
- subsequent moderate-training runs can be evaluated exactly because the snapshot is now persistent

## Regression

Run:

python .\run_nagato_knowledge_gain_eval_v10126.py

Expected:

STATUS : NAGATO_KNOWLEDGE_GAIN_EVAL_PASS
