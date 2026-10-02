# eval_context_aware_fact_learning_v107.py
#
# LLM_TRY v10.7 regression:
# Context-aware facts preserve preconditions and avoid false conflicts across
# different explicit conditions.

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from chat import (
    append_fact_store,
    compose_context_fact_answer,
    context_conflict_candidates,
    pair_fingerprint,
    trained_facts,
)


def main() -> None:
    print("=" * 96)
    print(" LLM_TRY v10.7 Context-Aware Fact Learning Regression")
    print("=" * 96)

    passed = 0
    total = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, total
        total += 1
        passed += int(condition)
        print(f"[{'PASS' if condition else 'FAIL'}] {name} : {detail}")

    with tempfile.TemporaryDirectory(prefix="llm_try_v107_") as tmp:
        root = Path(tmp)
        store = root / "fact_store.jsonl"
        state = root / "state.json"

        q1 = "Xについて"
        a1 = "条件Aのとき、Xの色は赤である。"
        q2 = "Xについて"
        a2 = "条件Bのとき、Xの色は青である。"
        q3 = "Xについて"
        a3 = "条件Aのとき、Xの色は青である。"

        check("save-context-red", append_fact_store(store, q1, a1), a1)
        check("save-context-blue", append_fact_store(store, q2, a2), a2)
        check("save-same-context-blue", append_fact_store(store, q3, a3), a3)

        state.write_text(
            json.dumps(
                {
                    "version": "v1.6.2",
                    "trained_fingerprints": [
                        pair_fingerprint(q1, a1),
                        pair_fingerprint(q2, a2),
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        facts = trained_facts(store, state, "X")
        check("two-conditioned-facts", len(facts) == 2, repr(facts))

        rendered = compose_context_fact_answer("X", facts)
        check(
            "condition-preserving-answer",
            rendered
            == "条件Aのとき、Xの色は赤である。条件Bのとき、Xの色は青である。",
            rendered,
        )

        conflicts = context_conflict_candidates(facts)
        check(
            "different-conditions-not-conflict",
            len(conflicts) == 0,
            f"conflicts={len(conflicts)}",
        )

        state.write_text(
            json.dumps(
                {
                    "version": "v1.6.2",
                    "trained_fingerprints": [
                        pair_fingerprint(q1, a1),
                        pair_fingerprint(q2, a2),
                        pair_fingerprint(q3, a3),
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        facts = trained_facts(store, state, "X")
        conflicts = context_conflict_candidates(facts)

        check(
            "same-context-different-value-candidate",
            len(conflicts) == 1,
            f"conflicts={len(conflicts)}",
        )

        if conflicts:
            left, right = conflicts[0]
            detail = (
                f"{left['condition']}: {left['value']} vs {right['value']}"
            )
            check(
                "conflict-scope-preserved",
                left["condition"] == right["condition"] == "条件A",
                detail,
            )

    print()
    print("Summary")
    print("-" * 96)
    print(f"Passed            : {passed}/{total}")
    print(f"Failed            : {total - passed}/{total}")
    print("Regression status : " + ("PASS" if passed == total else "FAIL"))

    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
