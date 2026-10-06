#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.5 Nagato Candidate Gate regression."""

from __future__ import annotations

import tempfile
from pathlib import Path

from promote_nagato_candidate_v10125 import atomic_promote


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 100)
    print(" LLM_TRY v10.12.5 Nagato Candidate Preservation / Promotion Regression")
    print("=" * 100)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        production = root / "production.pt"
        candidate = root / "candidate.pt"

        production.write_text("OLD", encoding="utf-8")
        candidate.write_text("NEW", encoding="utf-8")

        atomic_promote(candidate, production)

        check(
            "promotion-replaces-production",
            production.read_text(encoding="utf-8") == "NEW",
        )
        check(
            "candidate-consumed-on-promotion",
            not candidate.exists(),
        )
        check(
            "promotion-backup-cleaned",
            not (root / "production.pt.nagato-promotion-backup").exists(),
        )

    print()
    print("Promotion transaction : PASS")
    print("STATUS                : NAGATO_CANDIDATE_PROMOTION_GATE_PASS")


if __name__ == "__main__":
    main()
