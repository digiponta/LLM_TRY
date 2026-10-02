# eval_unknown_teaching_loop_v102.py
#
# LLM_TRY v10.2 regression:
# Unknown -> knowledge queue -> trusted teaching -> queue resolution.
#
# This test intentionally does not load the language model.  It validates the
# control-plane learning loop added in v10.2.

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import tempfile

from chat import (
    pre_generation_unknown_concept,
    resolve_route_queue,
    route_resolution_action,
    validate_teaching_answer,
)


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    suffix = f" : {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    return condition


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            rows.append(json.loads(raw))
    return rows


def main() -> None:
    print("=" * 88)
    print(" LLM_TRY v10.2 Unknown -> Teaching -> Incremental Learning Regression")
    print("=" * 88)

    passed = 0
    total = 0

    def run(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, total
        total += 1
        if check(name, condition, detail):
            passed += 1

    with tempfile.TemporaryDirectory(prefix="llm_try_v102_") as tmp:
        root = Path(tmp)
        knowledge_queue = root / "knowledge_queue.jsonl"
        teaching_queue = root / "teaching_queue.jsonl"
        gate_review_queue = root / "gate_review_queue.jsonl"

        args = Namespace(
            knowledge_queue=str(knowledge_queue),
            teaching_queue=str(teaching_queue),
            gate_review_queue=str(gate_review_queue),
        )

        unknown, focus = pre_generation_unknown_concept("宇宙とは")
        run("unknown-precheck", unknown and focus == "宇宙", f"focus={focus}")

        known, known_focus = pre_generation_unknown_concept("AIとは")
        run(
            "known-preservation",
            not known,
            f"focus={known_focus or '-'}",
        )

        route = route_resolution_action(
            args=args,
            resolution="UNKNOWN_KNOWLEDGE",
            action="retrieve/teach",
            user_text="宇宙とは",
            candidate_answer="",
            reason="unknown concept: 宇宙",
        )
        rows = read_rows(knowledge_queue)
        run("knowledge-queue-created", len(rows) == 1, route)
        run(
            "knowledge-queue-pending",
            len(rows) == 1
            and rows[0].get("resolution") == "UNKNOWN_KNOWLEDGE"
            and rows[0].get("user") == "宇宙とは"
            and rows[0].get("status", "pending") == "pending",
        )

        duplicate_route = route_resolution_action(
            args=args,
            resolution="UNKNOWN_KNOWLEDGE",
            action="retrieve/teach",
            user_text="宇宙とは",
            candidate_answer="",
            reason="unknown concept: 宇宙",
        )
        duplicate_rows = read_rows(knowledge_queue)
        run(
            "queue-deduplication",
            len(duplicate_rows) == 1,
            duplicate_route,
        )

        teacher_answer = (
            "宇宙は、物質・エネルギー・時空を含む、"
            "観測可能な世界全体を指す。"
        )
        teaching_ok, teaching_reason = validate_teaching_answer(
            "宇宙とは",
            teacher_answer,
        )
        run("trusted-teaching-validation", teaching_ok, teaching_reason)

        resolved = resolve_route_queue(
            knowledge_queue,
            "宇宙とは",
            "UNKNOWN_KNOWLEDGE",
        )
        resolved_rows = read_rows(knowledge_queue)
        run("knowledge-queue-resolved-count", resolved == 1, f"resolved={resolved}")
        run(
            "knowledge-queue-resolved-state",
            len(resolved_rows) == 1
            and resolved_rows[0].get("status") == "resolved"
            and bool(resolved_rows[0].get("resolved_at")),
        )

    print()
    print("Summary")
    print("-" * 88)
    print(f"Passed            : {passed}/{total}")
    print(f"Failed            : {total - passed}/{total}")
    print(
        "Regression status : "
        + ("PASS" if passed == total else "FAIL")
    )


if __name__ == "__main__":
    main()
