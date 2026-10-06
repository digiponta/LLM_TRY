# LLM_TRY v10.12.12.1 — Full data-nagato Learning Pipeline

This release adds a one-command pipeline that learns data-nagato.txt through both raw continued pretraining and semantic proposition training.

## Pipeline

production checkpoint
-> timestamped pre-run baseline
-> moderate raw continued pretraining
-> raw knowledge-gain evaluation
-> corpus-to-semantic dataset build
-> role-balanced semantic training
-> final semantic/generalization evaluation

The production checkpoint is never overwritten by the pipeline.

## Command

python .\learn_data_nagato_v1012121.py

## Candidates

Raw candidate:

model/model-gpu-v1.6.2-online-nagato-candidate.pt

Final raw+semantic candidate:

model/model-gpu-v1.6.2-online-nagato-semantic-candidate.pt

## Evaluation policy

The raw knowledge-gain evaluator is informative. Its QA threshold may still fail even when corpus NLL improves; this does not stop the semantic stage.

The final semantic evaluator is mandatory and must pass:

- plain QA gain threshold
- semantic gain threshold
- semantic improved > regressed
- protected INTERNALIZED retention
- persona retention
- metadata validation

## Safety

No automatic promotion is performed.

Expected final status:

STATUS : DATA_NAGATO_FULL_LEARNING_PASS
