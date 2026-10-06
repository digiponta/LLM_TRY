#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression for v10.12.0 post-batch internalized verification override."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from chat import active_internalized_verification_record, pair_fingerprint
from internalized_verification_v10119 import upsert_unstable, mark_batch_retrain


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 104)
    print(" LLM_TRY v10.12.0 Post-Batch Internalized Verification Override Regression")
    print("=" * 104)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        log = root / "chat_history.jsonl"
        state = root / "chat_learning_state.json"
        verification = root / "verification.jsonl"

        rows = [
            {
                "user": "文学とは",
                "assistant": "文学は言語による芸術を探究する学問である。",
                "source": "chat-manual",
            },
            {
                "user": "架空装置とは",
                "assistant": "架空装置は高速であり低消費電力である。",
                "source": "chat-manual",
            },
        ]
        log.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

        fps = [
            pair_fingerprint(row["user"], row["assistant"])
            for row in rows
        ]
        state.write_text(
            json.dumps({"trained_fingerprints": fps}, ensure_ascii=False),
            encoding="utf-8",
        )

        for row, fp in zip(rows, fps):
            concept = row["user"].replace("とは", "")
            upsert_unstable(
                verification,
                concept=concept,
                question=row["user"],
                teacher_answer=row["assistant"],
                candidate_answer="bad",
                reason="runtime demo",
                semantic=0.9,
                lexical=0.1,
                required=0.1,
                contradiction=False,
                missing_terms=("required",),
                fingerprint=fp,
            )
        mark_batch_retrain(verification, set(fps))

        literature = active_internalized_verification_record(
            verification, log, state, "文学"
        )
        device = active_internalized_verification_record(
            verification, log, state, "架空装置"
        )
        absent = active_internalized_verification_record(
            verification, log, state, "宇宙"
        )

        check(
            "literature-active-repair-overrides-retrieval",
            literature is not None
            and literature.question == "文学とは",
        )
        check(
            "device-active-repair-overrides-retrieval",
            device is not None
            and device.question == "架空装置とは",
        )
        check(
            "unrelated-concept-not-overridden",
            absent is None,
        )

    print()
    print("STATUS : POST_BATCH_INTERNALIZED_VERIFICATION_OVERRIDE_PASS")


if __name__ == "__main__":
    main()
