# LLM_TRY v10.6 — Compositional Fact Learning

v10.6 adds a structured Fact Store for conservative composition of multiple trained facts about the same subject.

Example:

~~~text
/teach XはYである。
/train
/teach XはZである。
/train

Xとは
-> Xは、Yであり、Zである。
~~~

Safety properties:

- facts are extracted only from conservative `XはYである。` / `XはYです。` teaching forms;
- a fact is used for composition only if its exact teaching pair fingerprint is present in the trained state;
- teach-only facts are not exposed as learned knowledge;
- untrained facts are excluded from composition;
- existing LLM generation and Known/Unknown gating remain available as fallback.

New files:

- `data/fact_store.jsonl` (created at runtime)
- `eval_compositional_fact_learning_v106.py`

Run:

~~~powershell
python eval_compositional_fact_learning_v106.py
~~~
