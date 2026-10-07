#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.16 Modifier-to-Condition Normalization regression."""

from modifier_condition_normalization_v10160 import (
    normalize_modifier_condition,
    normalize_modifier_condition_text,
)


CASES = [
    ("高温のCPUは停止する", "CPUは、高温の場合、停止する", True),
    ("低温のCPUは正常に動作する", "CPUは、低温の場合、正常に動作する", True),
    ("高負荷のGPUは温度が上がる", "GPUは、高負荷の場合、温度が上がる", True),
    ("夜間の道路は暗い", "道路は、夜間の場合、暗い", True),
    ("雨の日の道路は滑る", "道路は、雨の日の場合、滑る", True),
    ("空腹の猫は鳴く", "猫は、空腹の場合、鳴く", True),
    ("実行時のCPUは命令を処理する", "CPUは、実行時の場合、命令を処理する", True),
    # Safety regressions: attribute / relation must remain unchanged.
    ("赤い車は速い", "赤い車は速い", False),
    ("日本の首都は東京である", "日本の首都は東京である", False),
    ("文学の分類は複雑である", "文学の分類は複雑である", False),
    # Idempotence.
    ("CPUは、高温の場合、停止する", "CPUは、高温の場合、停止する", False),
]


def main() -> int:
    print("=" * 92)
    print(" LLM_TRY v10.16 Modifier-to-Condition Normalization Regression")
    print("=" * 92)

    passed = 0
    for source, expected, expected_transformed in CASES:
        result = normalize_modifier_condition(source)
        ok = (
            result.normalized == expected
            and result.transformed == expected_transformed
            and normalize_modifier_condition_text(source) == expected
        )
        label = "PASS" if ok else "FAIL"
        print(
            f"[{label}] {source} -> {result.normalized} "
            f"(transformed={result.transformed}, reason={result.reason})"
        )
        if ok:
            passed += 1

    print("-" * 92)
    print(f"Passed : {passed}/{len(CASES)}")
    print(f"Failed : {len(CASES) - passed}/{len(CASES)}")
    status = (
        "MODIFIER_TO_CONDITION_V10160_PASS"
        if passed == len(CASES)
        else "MODIFIER_TO_CONDITION_V10160_FAIL"
    )
    print(f"STATUS : {status}")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
