#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.16.2 Slash-Command Normalization Guard regression."""

from chat import normalize_runtime_input

CASES = [
    ("/condteach 高温のCPUは停止する", "/condteach 高温のCPUは停止する"),
    ("/condteach 低温のCPUは正常に動作する", "/condteach 低温のCPUは正常に動作する"),
    ("/teachq CPUが高温のときは？ => CPUは停止する。", "/teachq CPUが高温のときは? => CPUは停止する。"),
    ("/promote CPU => CPUは計算装置である。", "/promote CPU => CPUは計算装置である。"),
    ("高温の場合CPUはどうなる？", "CPUは、高温の場合、どうなる"),
    ("CPUが高温のときは？", "CPUは、高温の場合、どうなる"),
    ("低温時のCPUは？", "CPUは、低温の場合、どうなる"),
    ("高温のCPUは停止する", "CPUは、高温の場合、停止する"),
]

def main() -> int:
    print("=" * 96)
    print(" LLM_TRY v10.16.2 Slash-Command Normalization Guard Regression")
    print("=" * 96)
    passed = 0
    for source, expected in CASES:
        actual = normalize_runtime_input(source)
        ok = actual == expected
        print(f"[{'PASS' if ok else 'FAIL'}] {source} -> {actual}")
        if ok:
            passed += 1
    print("-" * 96)
    print(f"Passed : {passed}/{len(CASES)}")
    print(f"Failed : {len(CASES) - passed}/{len(CASES)}")
    status = "COMMAND_NORMALIZATION_GUARD_V10162_PASS" if passed == len(CASES) else "COMMAND_NORMALIZATION_GUARD_V10162_FAIL"
    print(f"STATUS : {status}")
    return 0 if passed == len(CASES) else 1

if __name__ == "__main__":
    raise SystemExit(main())
