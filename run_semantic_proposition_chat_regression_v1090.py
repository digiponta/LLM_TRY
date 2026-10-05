#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.9.0 Semantic Proposition Chat Routing Regression."""

from __future__ import annotations

import tempfile
from pathlib import Path

from chat import semantic_proposition_lookup
from semantic_proposition_v1090 import add_statement


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 88)
    print(" LLM_TRY v10.9.0 Semantic Proposition Chat Routing Regression")
    print("=" * 88)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "semantic_propositions.jsonl"

        add_statement(path, "架空装置は高速である。")
        add_statement(path, "架空装置は低消費電力である。")

        hit = semantic_proposition_lookup("架空装置とは", path)
        check("definition-query-hit", hit is not None, str(hit))
        check(
            "definition-query-compose",
            hit == (
                "架空装置",
                "架空装置は、高速であり、低消費電力である。",
            ),
            str(hit),
        )

        hit_bare = semantic_proposition_lookup("架空装置", path)
        check("bare-query-hit", hit_bare is not None, str(hit_bare))

        miss = semantic_proposition_lookup("未知装置とは", path)
        check("unknown-subject-miss", miss is None, str(miss))

    print()
    print("Atomic proposition store : PASS")
    print("Chat lookup              : PASS")
    print("Composed answer           : PASS")
    print("Unknown isolation         : PASS")
    print("STATUS                    : PROPOSITION_CHAT_ROUTING_PASS")


if __name__ == "__main__":
    main()
