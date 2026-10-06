# LLM_TRY v10.12.4 — Moderate data-nagato Continued Pretraining

v10.12.4 adds a conservative raw-text adaptation path for data-nagato.txt.

## Goal

Learn a meaningful amount from data-nagato.txt without applying the earlier
aggressive multi-epoch full-corpus training recipe directly to the current
verified online checkpoint.

## Default training profile

- Base model: model/model-gpu-v1.6.2-online.pt
- Output: model/model-gpu-v1.6.2-online-nagato-candidate.pt
- Epochs: 1
- Learning rate: 3e-6
- Block size: 256
- Stride: 128
- Validation ratio: 0.10
- Batch size: 8
- Frozen transformer blocks: first 2

This remains a raw causal-language-model objective. It is intentionally weaker
than the earlier 5-epoch Nagato pretraining experiments.

## Metadata preservation

The current production checkpoint metadata is inherited by the candidate,
including trained_fingerprints used by Internalized Checkpoint Binding.

Additional metadata records the Nagato adaptation configuration.

## Safety

The production checkpoint is not overwritten. Training always writes a separate
candidate checkpoint by default.

Run:

python .\train_nagato_moderate_v10124.py

Then inspect the candidate with:

python .\chat.py --model .\model\model-gpu-v1.6.2-online-nagato-candidate.pt

Recommended checks:

- /internalized
- 量子センサー
- 数学
- 文学
- 架空装置
- あなたは誰ですか

If the candidate remains stable, it can become the input to a later controlled
promotion step.
