# LLM_TRY v10.16.3 — Automatic Conditional Candidate Capture

v10.16.3 extends conditional semantics from explicit `/condteach` input to automatic detection of declarative conditional knowledge.

## Runtime flow

~~~text
高温のCPUは停止する
        |
        v
conditional statement detection
        |
        v
candidate queue (untrusted)
        |
        +--> /condcandidates
        |
        +--> /condapprove N | all
        v
Conditional Semantic Store
        |
        +--> immediate deterministic retrieval
        |
        v
/sleep
        |
        v
trusted conditional-semantic-sleep pair
        |
        v
existing incremental trainer
        |
        v
LLM internalization
~~~

## Safety rule

Auto-detected statements are **not trusted automatically**. They enter a pending candidate queue and require explicit approval before becoming retrievable knowledge or `/sleep` training material.

## Commands

~~~text
/condcandidates
/condapprove 1
/condapprove all
/conds
/sleep
~~~

## Verification

~~~powershell
python .\verify_conditional_candidate_v10163.py
~~~

Expected:

~~~text
STATUS : CONDITIONAL_CANDIDATE_LIFECYCLE_V10163_PASS
~~~
