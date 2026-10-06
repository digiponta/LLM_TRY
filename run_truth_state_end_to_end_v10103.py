#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.10.3 Truth State End-to-End Regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from internalized_knowledge_v10100 import pair_fingerprint
from knowledge_state_dispatcher_v10101 import dispatch_knowledge_state
from knowledge_state_resolver_v10100 import resolve_knowledge_state
from semantic_proposition_v1090 import add_statement
from subject_keyed_proposition_v1090 import sync_subject_index
from truth_aware_dispatch_v10103 import apply_truth_policy
from truth_state_v10103 import (
    effective_truth_record,
    upsert_truth_record,
)
from typed_subject_proposition_v10100 import sync_typed_index


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def resolve_dispatch_truth(query: str, common: dict, truth_path: Path):
    state = resolve_knowledge_state(query, **common)
    dispatch = dispatch_knowledge_state(state)
    truth = None
    result = None
    if dispatch.focus and dispatch.state != "UNKNOWN":
        truth = effective_truth_record(truth_path, dispatch.focus)
        result = apply_truth_policy(dispatch, truth)
        dispatch = result.dispatch
    return state, dispatch, truth, result


def main() -> None:
    print("=" * 100)
    print(" LLM_TRY v10.10.3 Truth State End-to-End Regression")
    print("=" * 100)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        props = root / "atomic.jsonl"
        subject = root / "subject.jsonl"
        typed = root / "typed.jsonl"
        unified = root / "unified.jsonl"
        log = root / "chat_history.jsonl"
        state_file = root / "chat_learning_state.json"
        corpus = root / "corpus.txt"
        truth_path = root / "truth.jsonl"

        add_statement(props, "GPUは高速である。")
        sync_subject_index(props, subject)
        sync_typed_index(props, typed)

        unified.write_text(
            json.dumps(
                {
                    "concept": "宇宙",
                    "assistant": "宇宙は、時空と物質を含む全体である。",
                    "source": "atomic-proposition",
                },
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

        iq = "量子センサーとは"
        ia = "量子センサーは、高感度計測を行うセンサーである。"
        fp = pair_fingerprint(iq, ia)
        log.write_text(
            json.dumps(
                {
                    "user": iq,
                    "assistant": ia,
                    "source": "chat-manual",
                },
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        state_file.write_text(
            json.dumps(
                {"trained_fingerprints": [fp]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        corpus.write_text(
            "時間について考える。時間とは何か。時間の流れ。\n",
            encoding="utf-8",
        )

        common = dict(
            typed_index_path=typed,
            subject_index_path=subject,
            unified_path=unified,
            learning_log=log,
            learning_state=state_file,
            raw_corpus_path=corpus,
            canonical_definitions={
                "cpu": "CPUは、命令を実行する中央処理装置である。",
            },
        )

        upsert_truth_record(
            truth_path,
            "GPU",
            "TRUE",
            source="test",
        )
        upsert_truth_record(
            truth_path,
            "宇宙",
            "CONTESTED",
            reason="複数の解釈がある",
            source="test",
        )
        upsert_truth_record(
            truth_path,
            "量子センサー",
            "FALSE",
            correction="量子センサーは、量子効果を利用して高感度計測を行う装置である。",
            source="test",
        )

        _, gpu_dispatch, gpu_truth, gpu_result = resolve_dispatch_truth(
            "GPUの性質は", common, truth_path
        )
        check(
            "true-retrieval",
            gpu_truth is not None
            and gpu_truth.state == "TRUE"
            and gpu_dispatch.action == "RETRIEVE"
            and not gpu_result.warning,
            str((gpu_dispatch, gpu_truth, gpu_result)),
        )

        _, cpu_dispatch, cpu_truth, cpu_result = resolve_dispatch_truth(
            "CPUとは", common, truth_path
        )
        check(
            "default-unverified",
            cpu_truth is not None
            and cpu_truth.state == "UNVERIFIED"
            and cpu_dispatch.action == "RETRIEVE"
            and "unverified" in cpu_result.warning,
            str((cpu_dispatch, cpu_truth, cpu_result)),
        )

        _, universe_dispatch, universe_truth, universe_result = (
            resolve_dispatch_truth("宇宙とは", common, truth_path)
        )
        check(
            "contested-warning",
            universe_truth is not None
            and universe_truth.state == "CONTESTED"
            and universe_dispatch.action == "RETRIEVE"
            and "contested" in universe_result.warning,
            str((universe_dispatch, universe_truth, universe_result)),
        )

        _, quantum_dispatch, quantum_truth, quantum_result = (
            resolve_dispatch_truth("量子センサーとは", common, truth_path)
        )
        check(
            "false-correction-overrides-generation",
            quantum_truth is not None
            and quantum_truth.state == "FALSE"
            and quantum_dispatch.action == "RETRIEVE"
            and quantum_result.correction_applied
            and "量子効果" in quantum_dispatch.answer,
            str((quantum_dispatch, quantum_truth, quantum_result)),
        )

        _, raw_dispatch, raw_truth, raw_result = resolve_dispatch_truth(
            "時間とは", common, truth_path
        )
        check(
            "raw-remains-blocked",
            raw_truth is not None
            and raw_truth.state == "UNVERIFIED"
            and raw_dispatch.action == "BLOCK",
            str((raw_dispatch, raw_truth, raw_result)),
        )

    print()
    print("TRUE route             : PASS")
    print("Default UNVERIFIED     : PASS")
    print("CONTESTED warning      : PASS")
    print("FALSE correction       : PASS")
    print("RAW block preservation : PASS")
    print("STATUS                 : TRUTH_STATE_END_TO_END_PASS")


if __name__ == "__main__":
    main()
