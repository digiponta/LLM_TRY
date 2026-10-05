# LLM_TRY v10.9.0 — Semantic Integration

This branch begins integration of validated LLM_SEM semantic mechanisms into
LLM_TRY without replacing the existing adaptive-learning or Known/Unknown
control plane.

## Phase 1: Semantic Proposition Compose / Decompose

The first integrated component is a bidirectional atomic proposition layer.

~~~text
XはYである。
XはZである。
    ↓ decompose
(X, Y)
(X, Z)
    ↓ compose
Xは、Yであり、Zである。
~~~

The reverse direction is also supported:

~~~text
Xは、Yであり、Zである。
    ↓
(X, Y)
(X, Z)
~~~

### Files

- `semantic_proposition_v1090.py`
- `run_semantic_proposition_integration_v1090.py`
- `run_semantic_proposition_chat_regression_v1090.py`

### Chat integration

`chat.py` adds:

~~~text
/propteach <statement>
/prop <subject>
/props
~~~

Normal concept queries now check the atomic proposition store before the
existing unified semantic-memory lookup and before model generation.

Example:

~~~text
/propteach 架空装置は高速である。
/propteach 架空装置は低消費電力である。
架空装置とは
~~~

Expected answer:

~~~text
架空装置は、高速であり、低消費電力である。
~~~

This path is retrieval/composition, not model generation.

## Design constraint

The existing LLM_TRY mechanisms remain unchanged in this phase:

- adaptive training
- Known/Unknown gate
- teaching queues
- unified semantic memory
- relation/fact behavior
- online checkpoint selection

The proposition layer is intentionally isolated so it can be regression-tested
before deeper integration with semantic routing and model internalization.

## Verification

~~~powershell
python .\run_semantic_proposition_integration_v1090.py
python .\run_semantic_proposition_chat_regression_v1090.py
python .\chat.py
~~~

Phase 2 will connect proposition normalization with the existing unified
semantic-memory/relation pipeline. Later phases can add LLM_SEM-style semantic
routing and controlled internalization.
