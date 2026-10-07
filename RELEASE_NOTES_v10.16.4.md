# LLM_TRY v10.16.4 — Raw-Input Conditional Candidate Guard

v10.16.4 fixes a false candidate-capture bug in v10.16.3.

## Observed bug

Conditional questions were normalized before candidate detection:

~~~text
CPUが高温のときは？
    -> CPUは、高温の場合、どうなる
    -> incorrectly captured as predicate="どうなる"
~~~

Already-approved facts could also be queued again as candidates.

## Fix

1. Preserve `raw_user_text` before semantic normalization.
2. Detect conditional questions on raw input and exclude them from candidate capture.
3. Parse declarative candidate statements from raw input only.
4. Skip candidates already present in the approved Conditional Semantic Store.
5. Existing approved facts continue through normal runtime processing instead of being swallowed by candidate capture.

## Verification

~~~powershell
python .\verify_conditional_candidate_guard_v10164.py
python .\verify_conditional_candidate_v10163.py
python .\verify_conditional_semantic_v10161.py
~~~

Expected new status:

~~~text
STATUS : RAW_CONDITIONAL_CANDIDATE_GUARD_V10164_PASS
~~~
