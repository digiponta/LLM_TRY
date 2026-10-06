#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.11.0 Semantic Knowledge Compatibility Regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from knowledge_state_resolver_v10100 import resolve_knowledge_state
from knowledge_state_dispatcher_v10101 import dispatch_knowledge_state
from semantic_knowledge_architecture_v10110 import (
    SemanticKnowledgeArchitecture,
    SemanticKnowledgeConfig,
)
from truth_state_v10103 import effective_truth_record
from truth_aware_dispatch_v10103 import apply_truth_policy


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (
        f" : {detail}" if detail else ""
    ))
    if not ok:
        raise AssertionError(name)


def legacy_resolve(query: str, config: SemanticKnowledgeConfig):
    state = resolve_knowledge_state(
        query,
        typed_index_path=config.typed_index_path,
        subject_index_path=config.subject_index_path,
        unified_path=config.unified_path,
        learning_log=config.learning_log,
        learning_state=config.learning_state,
        raw_corpus_path=config.raw_corpus_path,
        canonical_definitions=config.canonical_definitions,
    )
    dispatch = dispatch_knowledge_state(state)
    truth = None
    truth_result = None
    if dispatch.focus and dispatch.state != "UNKNOWN":
        truth = effective_truth_record(
            config.truth_store_path,
            dispatch.focus,
        )
        truth_result = apply_truth_policy(dispatch, truth)
        dispatch = truth_result.dispatch
    return state, dispatch, truth, truth_result


def main() -> None:
    print("=" * 104)
    print(" LLM_TRY v10.11.0 Semantic Knowledge Compatibility Regression")
    print("=" * 104)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        config = SemanticKnowledgeConfig(
            proposition_path=root / "atomic.jsonl",
            subject_index_path=root / "subject.jsonl",
            typed_index_path=root / "typed.jsonl",
            unified_path=root / "unified.jsonl",
            learning_log=root / "chat_history.jsonl",
            learning_state=root / "chat_learning_state.json",
            raw_corpus_path=root / "corpus.txt",
            truth_store_path=root / "truth.jsonl",
            canonical_definitions={
                "cpu": "CPUは、命令を実行する中央処理装置である。",
            },
        )
        architecture = SemanticKnowledgeArchitecture(config)

        architecture.teach_proposition("GPUは高速である。")
        config.unified_path.write_text(
            config.unified_path.read_text(encoding="utf-8")
            + json.dumps(
                {
                    "concept": "宇宙",
                    "assistant": "宇宙は全体である。",
                    "source": "compat-test",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        config.raw_corpus_path.write_text(
            "時間。時間。時間。\n",
            encoding="utf-8",
        )

        queries = (
            "GPUの性質は",
            "CPUとは",
            "宇宙とは",
            "時間とは",
            "未学習架空概念とは",
            "こんにちは",
        )

        for query in queries:
            old_state, old_dispatch, old_truth, old_truth_result = (
                legacy_resolve(query, config)
            )
            new = architecture.resolve(query)
            check(
                f"state:{query}",
                new.knowledge_state.state == old_state.state,
                f"old={old_state}, new={new.knowledge_state}",
            )
            check(
                f"action:{query}",
                new.dispatch.action == old_dispatch.action
                and new.dispatch.route == old_dispatch.route
                and new.dispatch.answer == old_dispatch.answer,
                f"old={old_dispatch}, new={new.dispatch}",
            )
            old_truth_state = old_truth.state if old_truth else ""
            check(
                f"truth:{query}",
                new.truth_state == old_truth_state,
                f"old={old_truth_state}, new={new.truth_state}",
            )

    print()
    print("Legacy resolver parity    : PASS")
    print("Legacy dispatcher parity  : PASS")
    print("Truth overlay parity      : PASS")
    print("STATUS                    : SEMANTIC_KNOWLEDGE_COMPATIBILITY_PASS")


if __name__ == "__main__":
    main()
