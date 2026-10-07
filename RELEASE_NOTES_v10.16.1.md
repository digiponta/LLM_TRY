# LLM_TRY v10.16.1 — Conditional Semantic Proposition

v10.16.1 extends v10.16 from declarative normalization to condition-aware semantic storage and retrieval.

## Fixed behavior

~~~text
高温の場合CPUはどうなる？
CPUが高温のときは？
低温時のCPUは？
~~~

These are normalized to `(subject, condition)` and can be answered without LLM generation when matching conditional knowledge exists.

## Structured knowledge

~~~text
高温のCPUは停止する
  -> subject=CPU
     condition=高温
     predicate=停止する
~~~

Teach with:

~~~text
/condteach 高温のCPUは停止する
/condteach 低温のCPUは正常に動作する
/conds
~~~

Then ordinary queries retrieve deterministically before the general semantic/LLM path.

## Verification

~~~powershell
python .\verify_conditional_semantic_v10161.py
~~~

Expected:

~~~text
STATUS : CONDITIONAL_SEMANTIC_V10161_PASS
~~~
