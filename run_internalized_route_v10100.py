#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.10.0 Internalized Knowledge Route Regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from chat import (
    pre_generation_unknown_concept,
    trained_known_concepts,
)
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
    print(" LLM_TRY v10.10.0 Internalized Knowledge Route Regression")
    print("=" * 96)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        log = root / "chat_history.jsonl"
        state = root / "chat_learning_state.json"

        quantum_q = "量子センサーとは"
        quantum_a = (
            "量子センサーは、量子的性質を利用して"
            "高感度計測を行うセンサーである。"
        )
        blackhole_q = "ブラックホールとは"
        blackhole_a = "ブラックホールは、非常に強い重力を持つ天体である。"

        rows = [
            {"user": quantum_q, "assistant": quantum_a, "source": "chat-manual"},
            {"user": blackhole_q, "assistant": blackhole_a, "source": "chat-manual"},
        ]
        log.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

        state.write_text(
            json.dumps(
                {
                    "version": "v10.10.0",
                    "trained_fingerprints": [
                        pair_fingerprint(quantum_q, quantum_a),
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        records = load_internalized_records(log, state)
        check("registry-one-item", len(records) == 1, str(records))

        quantum = internalized_record_for_focus(records, "量子センサー")
        check("quantum-route-present", quantum is not None, str(quantum))

        promoted = trained_known_concepts(log, state)
        check(
            "trained-concept-promoted",
            "量子センサー" in promoted,
            str(promoted),
        )

        quantum_unknown, quantum_focus = pre_generation_unknown_concept(
            quantum_q,
            promoted_concepts=promoted,
        )
        check(
            "internalized-bypasses-unknown",
            (not quantum_unknown) and quantum_focus == "量子センサー",
            f"{quantum_unknown=}, {quantum_focus=}",
        )

        blackhole = internalized_record_for_focus(records, "ブラックホール")
        check("untrained-route-absent", blackhole is None, str(blackhole))

        blackhole_unknown, blackhole_focus = pre_generation_unknown_concept(
            blackhole_q,
            promoted_concepts=promoted,
        )
        check(
            "untrained-stays-unknown",
            blackhole_unknown and blackhole_focus == "ブラックホール",
            f"{blackhole_unknown=}, {blackhole_focus=}",
        )

    print()
    print("Training-state proof       : PASS")
    print("Internalized route detect  : PASS")
    print("Unknown bypass             : PASS")
    print("Untrained isolation        : PASS")
    print("STATUS                     : INTERNALIZED_ROUTE_PASS")


if __name__ == "__main__":
    main()
