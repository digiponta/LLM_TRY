# LLM_TRY v10.16.5 — Conditional Semantic Runtime Stable

v10.16.5 consolidates the v10.16.0-v10.16.4 conditional-semantic work into a stable candidate.

## Stable capability

~~~text
modifier + subject + predicate
        |
        v
Modifier-to-Condition Normalization
        |
        +-- conditional question -> Query Plane
        |        |
        |        v
        |   (subject, condition)
        |        |
        |        v
        |   Conditional Retrieval
        |
        +-- declarative statement -> Candidate Plane
                 |
                 v
            Human Approval
                 |
                 v
        Conditional Semantic Store
                 |
                 +--> deterministic runtime retrieval
                 |
                 v
               /sleep
                 |
                 v
          LLM internalization
~~~

## Stable safety boundaries

- Slash commands bypass natural-language normalization.
- Conditional questions never enter candidate capture.
- Auto-detected declarative facts require explicit approval.
- Already stored facts are not queued again.
- Attribute/relation phrases are preserved unless proven conditional.
- Conditional retrieval returns stored propositions without LLM generation.

## Runtime status

The chat banner is now:

~~~text
LLM_TRY Chat - v10.16.5 Conditional Semantic Runtime Stable
~~~

`/semstatus` additionally reports:

~~~text
conditional : <path> (stored=N, candidates=M)
cond sleep  : internalized=I, pending=P
~~~

## Stable regression

Run:

~~~powershell
python .\verify_conditional_runtime_stable_v10165.py
~~~

The integrated verifier executes all five established conditional regressions (48 historical cases total) plus the new stable status integration check.

Expected:

~~~text
STATUS : CONDITIONAL_SEMANTIC_RUNTIME_STABLE_V10165_PASS
~~~

## Release boundary

v10.16.5 is intended as the final stable candidate before merging the conditional-semantic series into `main`.
