# LLM_TRY v10.12.11.1 — Timestamped Nagato Baseline Snapshots

v10.12.11.1 changes moderate data-nagato retraining so every run captures the exact production checkpoint that existed immediately before that run.

## Why

The historical fixed baseline:

model/model-gpu-v1.6.2-online-pre-nagato.pt

was intentionally preserved and not overwritten. That is useful for long-term comparison, but it is not sufficient for measuring the incremental gain of each later retraining run.

## New per-run baseline

Before every execution of:

python .\train_nagato_moderate_v10124.py

the current base checkpoint is copied to:

model/baselines/model-gpu-v1.6.2-online-pre-nagato-YYYYMMDD-HHMMSS-ffffff.pt

Microseconds are included to avoid name collisions.

Each snapshot is immutable because a new filename is generated on every run.

## Latest manifest

The trainer also writes:

model/baselines/nagato_baseline_latest.json

The manifest records:

- version
- creation time
- base model
- timestamped snapshot
- candidate output
- training corpus

## Compatibility path

The old fixed path is retained for compatibility:

model/model-gpu-v1.6.2-online-pre-nagato.pt

It is created only if absent and is never automatically overwritten.

## Knowledge-gain evaluation

evaluate_nagato_knowledge_gain_v10126.py now resolves --before as follows:

1. Explicit --before path, when supplied
2. Latest timestamped snapshot from nagato_baseline_latest.json
3. Compatibility baseline only if no manifest exists

Therefore the normal incremental comparison is now:

timestamped pre-run production
    ->
new Nagato candidate

Example:

python .\train_nagato_moderate_v10124.py

python .\evaluate_nagato_knowledge_gain_v10126.py --after .\model\model-gpu-v1.6.2-online-nagato-candidate.pt

No --before argument is needed.

## Regression

python .\run_timestamped_nagato_baseline_v1012111.py

Expected:

STATUS : TIMESTAMPED_NAGATO_BASELINE_PASS

## Verification

python .\verify_timestamped_nagato_baseline_v1012111.py

Expected:

STATUS : TIMESTAMPED_NAGATO_BASELINE_FULL_PASS
