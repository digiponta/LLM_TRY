#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.10.0 Internalized Knowledge Registry Regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from internalized_knowledge_v10100 import (
    internalized_record_for_focus,
    load_internalized_records,
    pair_fingerprint,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 96)
    print(" LLM_TRY v10.10.0 Internalized Knowledge Registry Regression")
    print("=" * 96)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        log = root / "chat_history.jsonl"
        state = root / "chat_learning_state.json"

        q1 = "量子センサーとは"
        a1 = "量子センサーは、量子的性質を利用して高感度計測を行うセンサーである。"
        q2 = "ブラックホールとは"
        a2 = "ブラックホールは、非常に強い重力を持つ天体である。"

        rows = [
            {
                "user": q1,
                "assistant": a1,
                "source": "chat-manual",
            },
            {
                "user": q2,
                "assistant": a2,
                "source": "chat-manual",
            },
        ]
        log.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

        # Only the quantum-sensor pair has actually been consumed by /train.
        state.write_text(
            json.dumps(
                {
                    "version": "v10.10.0",
                    "trained_fingerprints": [
                        pair_fingerprint(q1, a1),
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        records = load_internalized_records(log, state)
        check("internalized-count", len(records) == 1, str(records))

        quantum = internalized_record_for_focus(records, "量子センサー")
        check("quantum-internalized", quantum is not None, str(quantum))
        check(
            "canonical-question-preserved",
            quantum is not None and quantum.question == q1,
            str(quantum),
        )

        blackhole = internalized_record_for_focus(records, "ブラックホール")
        check(
            "untrained-not-internalized",
            blackhole is None,
            str(blackhole),
        )

    print()
    print("Trusted pair filter       : PASS")
    print("Training-state proof      : PASS")
    print("Internalized concept map  : PASS")
    print("Untrained isolation       : PASS")
    print("STATUS                    : INTERNALIZED_REGISTRY_PASS")


if __name__ == "__main__":
    main()
