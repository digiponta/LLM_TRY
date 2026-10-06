#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.2 Candidate Checkpoint / Rollback Regression."""

from __future__ import annotations

import tempfile
from pathlib import Path

from chat import (
    batch_candidate_paths,
    discard_batch_candidate,
    promote_batch_candidate,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 108)
    print(" LLM_TRY v10.12.2 Candidate Checkpoint / Rollback Regression")
    print("=" * 108)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        production_ckpt = root / "model-online.pt"
        production_state = root / "learning_state.json"

        candidate_ckpt, candidate_state = batch_candidate_paths(
            production_ckpt,
            production_state,
        )
        check(
            "candidate-paths-separated",
            candidate_ckpt != production_ckpt
            and candidate_state != production_state
            and ".candidate" in candidate_ckpt.name
            and ".candidate" in candidate_state.name,
            f"{candidate_ckpt.name}, {candidate_state.name}",
        )

        production_ckpt.write_text("OLD_CHECKPOINT", encoding="utf-8")
        production_state.write_text("OLD_STATE", encoding="utf-8")
        candidate_ckpt.write_text("BAD_CANDIDATE", encoding="utf-8")
        candidate_state.write_text("BAD_STATE", encoding="utf-8")

        discard_batch_candidate(candidate_ckpt, candidate_state)
        check(
            "rollback-keeps-production",
            production_ckpt.read_text(encoding="utf-8") == "OLD_CHECKPOINT"
            and production_state.read_text(encoding="utf-8") == "OLD_STATE"
            and not candidate_ckpt.exists()
            and not candidate_state.exists(),
        )

        candidate_ckpt.write_text("GOOD_CANDIDATE", encoding="utf-8")
        candidate_state.write_text("GOOD_STATE", encoding="utf-8")
        promote_batch_candidate(
            candidate_ckpt,
            candidate_state,
            production_ckpt,
            production_state,
        )
        check(
            "pass-promotes-checkpoint-and-state",
            production_ckpt.read_text(encoding="utf-8") == "GOOD_CANDIDATE"
            and production_state.read_text(encoding="utf-8") == "GOOD_STATE"
            and not candidate_ckpt.exists()
            and not candidate_state.exists(),
        )

        leftovers = list(root.glob("*.promotion-backup"))
        check(
            "promotion-backups-cleaned",
            not leftovers,
            str(leftovers),
        )

    print()
    print("Candidate isolation : PASS")
    print("Rollback retention  : PASS")
    print("Dual-file promotion : PASS")
    print("STATUS              : BATCH_REPAIR_CANDIDATE_ROLLBACK_PASS")


if __name__ == "__main__":
    main()
