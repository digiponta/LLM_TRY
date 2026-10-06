#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare a deterministic v10.12.0 multi-item batch-repair runtime demo.

This script does not train or alter model checkpoint weights.
It:
  1. finds trusted CURRENT internalized records for selected concepts,
  2. creates verification repair tasks,
  3. removes only those fingerprints from chat_learning_state.json so
     online_train.py will treat them as pending/rebind,
  4. marks the tasks as retrain.

Then launch chat.py and use /batchstatus -> /trainbatch -> re-query concepts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from internalized_knowledge_v10100 import (
    internalized_record_for_focus,
    load_internalized_records,
)
from internalized_verification_v10119 import (
    mark_batch_retrain,
    upsert_unstable,
    verification_summary,
)


DEFAULT_CONCEPTS = ("文学", "架空装置")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare deterministic multi-item verification repair tasks "
            "for the v10.12.0 runtime demonstration."
        )
    )
    parser.add_argument(
        "--concept",
        action="append",
        dest="concepts",
        help=(
            "Internalized concept to stage. Repeat for multiple concepts. "
            "Default: 文学, 架空装置"
        ),
    )
    parser.add_argument(
        "--learning-log",
        default="data/chat_history.jsonl",
    )
    parser.add_argument(
        "--learning-state",
        default="data/chat_learning_state.json",
    )
    parser.add_argument(
        "--verification",
        default="data/internalized_verification_v10119.jsonl",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the plan without changing any files.",
    )
    return parser.parse_args()


def load_state(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Learning state not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fingerprints = payload.get("trained_fingerprints")
    if not isinstance(fingerprints, list):
        raise ValueError(
            "learning state must contain trained_fingerprints list"
        )
    return payload


def backup_file(path: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(path.name + f".v10120-demo-{stamp}.bak")
    shutil.copy2(path, backup)
    return backup


def main() -> None:
    args = parse_args()
    concepts = tuple(args.concepts or DEFAULT_CONCEPTS)
    learning_log = Path(args.learning_log)
    learning_state = Path(args.learning_state)
    verification = Path(args.verification)

    state = load_state(learning_state)
    trained = {
        str(value)
        for value in state.get("trained_fingerprints", [])
    }

    records = load_internalized_records(
        learning_log,
        learning_state,
    )

    selected = []
    missing = []
    for concept in concepts:
        record = internalized_record_for_focus(records, concept)
        if record is None:
            missing.append(concept)
        else:
            selected.append(record)

    print("=" * 104)
    print(" LLM_TRY v10.12.0 Multi-Item Batch Repair Runtime Demo Preparation")
    print("=" * 104)
    print("Concepts requested :", ", ".join(concepts))
    print("Selected CURRENT   :", len(selected))
    print("Missing/not current:", ", ".join(missing) if missing else "-")
    print("Dry run            :", args.dry_run)
    print()

    for index, record in enumerate(selected, 1):
        print(
            f"{index:02d}. concept={record.concept!r} "
            f"question={record.question!r} "
            f"fingerprint={record.fingerprint[:12]}..."
        )
        print(f"    teacher={record.teacher_answer}")

    if len(selected) < 2:
        raise SystemExit(
            "Need at least two CURRENT internalized concepts for "
            "the multi-item runtime demonstration."
        )

    if args.dry_run:
        print()
        print("DRY RUN: no files changed.")
        return

    backup = backup_file(learning_state)

    staged_fingerprints: set[str] = set()
    for record in selected:
        # The task is deliberately synthetic only at preparation time.
        # Actual /trainbatch and post-training verification use the real
        # trusted teacher pair and real model generation.
        upsert_unstable(
            verification,
            concept=record.concept,
            question=record.question,
            teacher_answer=record.teacher_answer,
            candidate_answer=(
                f"{record.concept}は、"
                "v10.12.0 runtime demo 用の不完全な候補回答である。"
            ),
            reason=(
                "v10.12.0 deterministic runtime-demo repair task; "
                "actual repair training and re-verification are live"
            ),
            semantic=0.999,
            lexical=0.0,
            required=0.0,
            contradiction=False,
            missing_terms=("runtime-demo-required-content",),
            fingerprint=record.fingerprint,
        )
        staged_fingerprints.add(record.fingerprint)
        trained.discard(record.fingerprint)

    state["trained_fingerprints"] = sorted(trained)
    state["version"] = state.get("version", "v1.6.2")
    learning_state.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    changed = mark_batch_retrain(
        verification,
        staged_fingerprints,
    )
    summary = verification_summary(verification)

    print()
    print("Preparation completed.")
    print("Learning-state backup:", backup)
    print("Staged tasks         :", len(staged_fingerprints))
    print("New retrain changes  :", changed)
    print(
        "Verification summary : "
        f"pending={summary.pending} "
        f"retrain={summary.retrain} "
        f"verified={summary.verified} "
        f"failed={summary.failed} "
        f"total={summary.total}"
    )
    print()
    print("Next:")
    print("  1. start chat.py with the CURRENT checkpoint")
    print("  2. /batchstatus")
    print("  3. /trainbatch")
    print("  4. re-query each staged concept")
    print("  5. /verifystatus")
    print()
    print("NOTE: checkpoint weights were not modified by this script.")


if __name__ == "__main__":
    main()
