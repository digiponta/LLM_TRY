#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.11.1 Timestamped Nagato baseline regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from evaluate_nagato_knowledge_gain_v10126 import (
    DEFAULT_BEFORE,
    resolve_before_path,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 116)
    print(" LLM_TRY v10.12.11.1 Timestamped Nagato Baseline Regression")
    print("=" * 116)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        baseline_dir = root / "baselines"
        baseline_dir.mkdir()

        snap1 = baseline_dir / "model-pre-nagato-20261006-150000-000001.pt"
        snap2 = baseline_dir / "model-pre-nagato-20261006-150000-000002.pt"
        snap1.write_bytes(b"baseline-1")
        snap2.write_bytes(b"baseline-2")

        check("timestamped-name-unique", snap1.name != snap2.name)

        manifest = baseline_dir / "nagato_baseline_latest.json"
        manifest.write_text(
            json.dumps(
                {
                    "version": "v10.12.11.1",
                    "snapshot": str(snap2),
                    "base_model": "model/model-gpu-v1.6.2-online.pt",
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        args = SimpleNamespace(
            before=DEFAULT_BEFORE,
            baseline_manifest=str(manifest),
        )
        resolved = resolve_before_path(args)
        check("manifest-latest-selected", resolved == snap2, str(resolved))

        explicit = SimpleNamespace(
            before=str(snap1),
            baseline_manifest=str(manifest),
        )
        resolved_explicit = resolve_before_path(explicit)
        check("explicit-before-wins", resolved_explicit == snap1, str(resolved_explicit))

    print()
    print("Timestamp uniqueness : PASS")
    print("Latest manifest      : PASS")
    print("Evaluator resolution : PASS")
    print("STATUS               : TIMESTAMPED_NAGATO_BASELINE_PASS")


if __name__ == "__main__":
    main()
