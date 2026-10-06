# LLM_TRY v10.12.7 — Knowledge Gain QA Probes

v10.12.7 extends Nagato knowledge-gain evaluation from raw-corpus NLL to source-grounded question answering.

## Goal

Measure whether moderate training on data/data-nagato.txt improves the model's ability to answer questions about corpus content, not only next-token prediction.

## Corpus-grounded probes

The repository now includes:

data/nagato_gain_probes_v10127.jsonl

Each probe records:

- probe_id
- concept
- question
- teacher answer
- exact source_text
- source path

The probes are used only to measure how faithfully the model learned the corpus. They are not treated as independently verified truth.

The initial benchmark contains 10 probes and excludes existing protected INTERNALIZED concepts such as 宇宙, 数学, 文学, 架空装置, and 量子センサー.

## Probe builder

Run:

python .\build_nagato_gain_probes_v10127.py

This scans the local Nagato corpus for simple source-grounded Japanese "Xは..." statements and creates a traceable QA probe file.

## Evaluation

The existing evaluator is extended for v10.12.7:

python .\evaluate_nagato_knowledge_gain_v10126.py ^
  --before .\model\model-gpu-v1.6.2-online-pre-nagato.pt ^
  --after  .\model\model-gpu-v1.6.2-online.pt

PowerShell:

python .\evaluate_nagato_knowledge_gain_v10126.py `
  --before .\model\model-gpu-v1.6.2-online-pre-nagato.pt `
  --after  .\model\model-gpu-v1.6.2-online.pt

The evaluator reports per-probe:

- before answer
- after answer
- composite QA score
- gain
- GAIN / SAME / REGRESS

Aggregate QA PASS requires:

- mean QA gain >= +0.01
- improved probes > regressed probes

The final summary now includes:

qa_gain=...
qa_status=PASS|FAIL

## Report

results/nagato_knowledge_gain_v10127.json

The JSON report includes source-grounded probe metadata and aggregate gain counts.

## Regression

Run:

python .\run_nagato_qa_probe_gain_v10127.py

Expected:

STATUS : NAGATO_QA_PROBE_GAIN_PASS
