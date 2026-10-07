#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.16.1 Conditional Semantic Proposition regression."""

from pathlib import Path
from tempfile import TemporaryDirectory

from conditional_semantic_v10161 import (
    add_conditional_statement,
    answer_conditional_query,
    parse_conditional_query,
    parse_conditional_statement,
)

def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" : {detail}" if detail else ""))
    return ok

def main() -> int:
    print("=" * 100)
    print(" LLM_TRY v10.16.1 Conditional Semantic Proposition Regression")
    print("=" * 100)
    passed = 0
    total = 0

    statement_cases = [
        ("高温のCPUは停止する", "CPU", "高温", "停止する"),
        ("低温のCPUは正常に動作する", "CPU", "低温", "正常に動作する"),
        ("高負荷のGPUは温度が上がる", "GPU", "高負荷", "温度が上がる"),
    ]
    for source, subject, condition, predicate in statement_cases:
        total += 1
        row = parse_conditional_statement(source)
        ok = row is not None and row.subject == subject and row.condition == condition and row.predicate == predicate
        passed += check(f"statement:{source}", ok, row.render() if row else "parse failed")

    query_cases = [
        ("高温の場合CPUはどうなる？", "CPU", "高温"),
        ("CPUが高温のときは？", "CPU", "高温"),
        ("低温時のCPUは？", "CPU", "低温"),
        ("CPUは高温の場合どうなる？", "CPU", "高温"),
    ]
    for source, subject, condition in query_cases:
        total += 1
        row = parse_conditional_query(source)
        ok = row.matched and row.subject == subject and row.condition == condition
        passed += check(f"query:{source}", ok, f"subject={row.subject!r} condition={row.condition!r} normalized={row.normalized!r}")

    for source in ("赤い車は速い？", "日本の首都は？", "CPUとは？"):
        total += 1
        row = parse_conditional_query(source)
        ok = not row.matched
        passed += check(f"safety:{source}", ok, row.reason)

    with TemporaryDirectory() as td:
        path = Path(td) / "conditional.jsonl"
        add_conditional_statement(path, "高温のCPUは停止する")
        add_conditional_statement(path, "低温のCPUは正常に動作する")
        retrieval_cases = [
            ("高温の場合CPUはどうなる？", "CPUは、高温の場合、停止する。"),
            ("CPUが高温のときは？", "CPUは、高温の場合、停止する。"),
            ("低温時のCPUは？", "CPUは、低温の場合、正常に動作する。"),
        ]
        for source, expected in retrieval_cases:
            total += 1
            parsed, answer = answer_conditional_query(path, source)
            ok = parsed.matched and answer == expected
            passed += check(f"retrieve:{source}", ok, answer or "MISS")

    print("-" * 100)
    print(f"Passed : {passed}/{total}")
    print(f"Failed : {total - passed}/{total}")
    status = "CONDITIONAL_SEMANTIC_V10161_PASS" if passed == total else "CONDITIONAL_SEMANTIC_V10161_FAIL"
    print(f"STATUS : {status}")
    return 0 if passed == total else 1

if __name__ == "__main__":
    raise SystemExit(main())
