# eval_compositional_fact_learning_v106.py
#
# LLM_TRY v10.6 regression:
# Multiple trained copular facts for the same subject are composed safely.

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from chat import (
    append_fact_store,
    compose_fact_answer,
    pair_fingerprint,
    trained_fact_values,
)


def main() -> None:
    print("=" * 92)
    print(" LLM_TRY v10.6.2 Single/Multi Fact Retrieval Regression")
    print("=" * 92)

    passed = 0
    total = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, total
        total += 1
        passed += int(condition)
        print(f"[{'PASS' if condition else 'FAIL'}] {name} : {detail}")

    with tempfile.TemporaryDirectory(prefix="llm_try_v106_") as tmp:
        root = Path(tmp)
        store = root / "fact_store.jsonl"
        state = root / "state.json"

        q1 = "Xとは"
        a1 = "XはYである。"
        q2 = "Xとは"
        a2 = "XはZである。"
        q3 = "Xとは"
        a3 = "XはWである。"

        check(
            "save-fact-Y",
            append_fact_store(store, q1, a1),
            "X -> Y",
        )
        check(
            "save-fact-Z",
            append_fact_store(store, q2, a2),
            "X -> Z",
        )
        check(
            "save-fact-W",
            append_fact_store(store, q3, a3),
            "X -> W (not trained)",
        )

        state.write_text(
            json.dumps(
                {
                    "version": "v1.6.2",
                    "trained_fingerprints": [
                        pair_fingerprint(q1, a1),
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        values = trained_fact_values(store, state, "X")
        check(
            "single-trained-fact-only",
            values == ["Y"],
            repr(values),
        )
        check(
            "single-fact-compose",
            compose_fact_answer("X", values) == "Xは、Yである。",
            compose_fact_answer("X", values),
        )
        check(
            "single-fact-runtime-eligible",
            len(values) >= 1,
            f"trained_fact_count={len(values)}",
        )

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

        values = trained_fact_values(store, state, "X")
        check(
            "two-trained-facts",
            values == ["Y", "Z"],
            repr(values),
        )

        composed = compose_fact_answer("X", values)
        check(
            "two-fact-composition",
            composed == "Xは、Yであり、Zである。",
            composed,
        )

        check(
            "untrained-fact-excluded",
            "W" not in values,
            repr(values),
        )

    print()
    print("Summary")
    print("-" * 92)
    print(f"Passed            : {passed}/{total}")
    print(f"Failed            : {total - passed}/{total}")
    print("Regression status : " + ("PASS" if passed == total else "FAIL"))

    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
