#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.6 Nagato Knowledge Gain Evaluation regression."""

from __future__ import annotations

from evaluate_nagato_knowledge_gain_v10126 import window_starts


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 104)
    print(" LLM_TRY v10.12.6 Nagato Knowledge Gain Evaluation Regression")
    print("=" * 104)

    starts = window_starts(
        length=1000,
        block_size=256,
        stride=128,
    )
    check(
        "window-starts-nonempty",
        bool(starts),
        str(starts),
    )
    check(
        "window-starts-begin-zero",
        starts[0] == 0,
        str(starts[0]),
    )
    check(
        "window-starts-cover-tail",
        starts[-1] == 1000 - 257,
        str(starts[-1]),
    )
    check(
        "window-starts-monotonic",
        all(a < b for a, b in zip(starts, starts[1:])),
    )

    print()
    print("Held-out window coverage : PASS")
    print("STATUS                   : NAGATO_KNOWLEDGE_GAIN_EVAL_PASS")


if __name__ == "__main__":
    main()
