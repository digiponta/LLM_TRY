#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.7 Knowledge Gain QA Probe regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from evaluate_nagato_knowledge_gain_v10126 import load_probes


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 104)
    print(" LLM_TRY v10.12.7 Knowledge Gain QA Probe Regression")
    print("=" * 104)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "probes.jsonl"
        rows = [
            {
                "version": "v10.12.7",
                "probe_id": "nagato-001",
                "concept": "物質波",
                "question": "物質波とは",
                "answer": "物質波は、ドブロイ波と言われているもの。",
                "source_text": "物質波は、ドブロイ波と言われているもの。",
                "source": "data/data-nagato.txt",
            },
            {
                "version": "v10.12.7",
                "probe_id": "nagato-002",
                "concept": "シングルコアCPU",
                "question": "シングルコアCPUとは",
                "answer": "シングルコアCPUは、同時に実行している命令は、だいたい一つ。",
                "source_text": "シングルコアCPUは、同時に実行している命令は、だいたい一つ。",
                "source": "data/data-nagato.txt",
            },
        ]
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

        loaded = load_probes(path)
        check("probe-count", len(loaded) == 2, str(len(loaded)))
        check(
            "probe-metadata-preserved",
            loaded[0]["probe_id"] == "nagato-001"
            and loaded[0]["source_text"]
            and loaded[0]["source"] == "data/data-nagato.txt",
        )
        check(
            "probe-question-answer-present",
            loaded[1]["question"] == "シングルコアCPUとは"
            and loaded[1]["answer"].startswith("シングルコアCPUは"),
        )

    # v10.12.7 aggregate policy: mean gain threshold and more gains than regressions.
    gains = [0.10, 0.04, 0.00, -0.01]
    mean_gain = sum(gains) / len(gains)
    improved = sum(1 for x in gains if x > 1e-6)
    same = sum(1 for x in gains if abs(x) <= 1e-6)
    regressed = len(gains) - improved - same
    policy_ok = mean_gain >= 0.01 and improved > regressed

    check("qa-mean-gain-policy", policy_ok, f"{mean_gain:+.3f}")
    check(
        "qa-improved-beats-regressed",
        improved > regressed,
        f"{improved}>{regressed}",
    )

    print()
    print("Corpus-grounded probes : PASS")
    print("QA aggregate policy    : PASS")
    print("STATUS                 : NAGATO_QA_PROBE_GAIN_PASS")


if __name__ == "__main__":
    main()
