# LLM_TRY v10.7 — Context-Aware Fact Learning

v10.7 extends the v10.6 Fact Store from unconditional facts to condition-aware propositions.

Representation:

~~~text
Subject + Relation + Value + Condition
~~~

Examples:

~~~text
条件Aのとき、Xの色は赤である。
条件Bのとき、Xの色は青である。
~~~

These facts coexist because their explicit conditions are different.

Conflict handling is conservative:

- same subject + same relation + same condition + different value -> conflict candidate;
- different explicit conditions -> coexist;
- domain-level value exclusivity is not inferred automatically;
- conflict candidates are not automatically deleted or overwritten.

Supported initial forms:

~~~text
XはYである。
条件Aのとき、XはYである。
Xの色は赤である。
条件Aのとき、Xの色は赤である。
~~~

Run:

~~~powershell
python eval_compositional_fact_learning_v106.py
python eval_context_aware_fact_learning_v107.py
~~~
