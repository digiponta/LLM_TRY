#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.9.0 Semantic Proposition Integration Regression."""

from __future__ import annotations

import tempfile
from pathlib import Path

from semantic_proposition_v1090 import (
    Proposition,
    add_statement,
    compose_propositions,
    compose_subject,
    decompose_statement,
    load_propositions,
    merge_propositions,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 88)
    print(" LLM_TRY v10.9.0 Semantic Proposition Integration Regression")
    print("=" * 88)

    check(
        "decompose-atomic",
        decompose_statement("XはYである。")
        == [Proposition("X", "Y")],
    )

    check(
        "decompose-compound",
        decompose_statement("Xは、Yであり、Zである。")
        == [Proposition("X", "Y"), Proposition("X", "Z")],
    )

    check(
        "compose-two",
        compose_propositions([
            Proposition("X", "Y"),
            Proposition("X", "Z"),
        ]) == "Xは、Yであり、Zである。",
    )

    original = "量子センサーは、高感度であり、量子的性質を利用する技術である。"
    check(
        "roundtrip",
        compose_propositions(decompose_statement(original)) == original,
    )

    merged = merge_propositions(
        [Proposition("X", "Y")],
        [Proposition("X", "Z"), Proposition("X", "Y")],
    )
    check(
        "merge-deduplicate",
        merged == [Proposition("X", "Y"), Proposition("X", "Z")],
    )

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "props.jsonl"
        add_statement(path, "架空装置は高速である。")
        add_statement(path, "架空装置は低消費電力である。")
        check("persistent-count", len(load_propositions(path)) == 2)
        check(
            "persistent-compose",
            compose_subject(path, "架空装置")
            == "架空装置は、高速であり、低消費電力である。",
        )

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "props.jsonl"
        added = add_statement(
            path,
            "架空装置は、高速であり、低消費電力である。",
        )
        check("compound-to-two-atoms", len(added) == 2)
        check("compound-store-two-atoms", len(load_propositions(path)) == 2)

    check(
        "unsupported-fails-closed",
        decompose_statement("Xとは何ですか") == [],
    )

    print()
    print("Compose                  : PASS")
    print("Decompose                : PASS")
    print("Bidirectional round-trip : PASS")
    print("Persistent merge         : PASS")
    print("STATUS                   : SEMANTIC_PROPOSITION_INTEGRATION_PASS")


if __name__ == "__main__":
    main()
