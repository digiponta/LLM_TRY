#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression for v10.13.2 modifier-aware subject canonicalization."""

from __future__ import annotations

from pathlib import Path

from build_nagato_gain_probes_v10127 import sentence_candidates


class Args:
    min_answer_chars = 8
    max_answer_chars = 160
    data = "data/data-nagato.txt"


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 104)
    print(" LLM_TRY v10.13.2 Modifier-Aware Subject Canonicalization Regression")
    print("=" * 104)

    sample = """重力波は時空の変動として伝播する。
強力な重力波は遠方まで伝播する可能性がある。
宇宙は多様な構造を含んでいる。
全体としての宇宙は一つの系として考えられる。
結果、宇宙は観測者の記述に依存する。"""

    rows = sentence_candidates(sample, Args())
    by_answer = {row["answer"]: row for row in rows}

    r = by_answer["強力な重力波は遠方まで伝播する可能性がある。"]
    check("modifier-subject", r["concept"] == "重力波", str(r))
    check("modifier-preserved", r["modifier"] == "強力な", str(r))
    check(
        "full-proposition-preserved",
        r["mapping"] == "重力波 => 強力な重力波は遠方まで伝播する可能性がある。",
        r["mapping"],
    )

    u = by_answer["全体としての宇宙は一つの系として考えられる。"]
    check("relative-modifier-subject", u["concept"] == "宇宙", str(u))
    check("relative-modifier-preserved", u["modifier"] == "全体としての", str(u))

    conservative = by_answer["結果、宇宙は観測者の記述に依存する。"]
    check(
        "comma-prefix-not-stripped",
        conservative["concept"] == "結果、宇宙",
        str(conservative),
    )

    corpus = Path(Args.data)
    if corpus.exists():
        actual = sentence_candidates(corpus.read_text(encoding="utf-8"), Args())
        canonicalized = [row for row in actual if row.get("modifier")]
        check("data-nagato-canonicalized", len(canonicalized) > 0, str(len(canonicalized)))
        print()
        print("data-nagato candidates   :", len(actual))
        print("canonicalized modifiers :", len(canonicalized))
        for row in canonicalized[:12]:
            print(" ", row["mapping"])

    print()
    print("STATUS : MODIFIER_SUBJECT_CANONICALIZATION_PASS")


if __name__ == "__main__":
    main()
