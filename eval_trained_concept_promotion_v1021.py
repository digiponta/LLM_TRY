# eval_trained_concept_promotion_v1021.py
#
# v10.2.1 control-plane regression:
# taught-only concept remains UNKNOWN; /train-consumed concept becomes KNOWN.

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from chat import (
    pair_fingerprint,
    pre_generation_unknown_concept,
    trained_known_concepts,
)


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    suffix = f" : {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    return condition


def main() -> None:
    print("=" * 88)
    print(" LLM_TRY v10.2.1 Trained Concept Promotion Regression")
    print("=" * 88)

    passed = 0
    total = 0

    def run(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, total
        total += 1
        if check(name, condition, detail):
            passed += 1

    with tempfile.TemporaryDirectory(prefix="llm_try_v1021_") as tmp:
        root = Path(tmp)
        log = root / "chat_history.jsonl"
        state = root / "chat_learning_state.json"

        question = "宇宙とは"
        answer = "宇宙は、物質・エネルギー・時空を含む世界全体を指す。"

        row = {
            "user": question,
            "assistant": answer,
            "source": "chat-manual",
        }
        log.write_text(
            json.dumps(row, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        state.write_text(
            json.dumps(
                {"version": "v1.6.2", "trained_fingerprints": []},
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        before_promoted = trained_known_concepts(log, state)
        before_unknown, focus = pre_generation_unknown_concept(
            question,
            promoted_concepts=before_promoted,
        )
        run(
            "teach-only-stays-unknown",
            before_unknown and focus == "宇宙",
            f"promoted={sorted(before_promoted)}",
        )

        fp = pair_fingerprint(question, answer)
        state.write_text(
            json.dumps(
                {"version": "v1.6.2", "trained_fingerprints": [fp]},
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        after_promoted = trained_known_concepts(log, state)
        after_unknown, after_focus = pre_generation_unknown_concept(
            question,
            promoted_concepts=after_promoted,
        )
        run(
            "trained-concept-promoted",
            "宇宙" in after_promoted,
            f"promoted={sorted(after_promoted)}",
        )
        run(
            "trained-concept-passes-precheck",
            not after_unknown and after_focus == "宇宙",
            f"focus={after_focus}",
        )

        static_unknown, static_focus = pre_generation_unknown_concept("ブラックホールとは")
        run(
            "untrained-other-concept-stays-unknown",
            static_unknown and static_focus == "ブラックホール",
            f"focus={static_focus}",
        )

        known_unknown, known_focus = pre_generation_unknown_concept("AIとは")
        run(
            "static-known-preserved",
            not known_unknown and known_focus == "AI",
            f"focus={known_focus}",
        )

    print()
    print("Summary")
    print("-" * 88)
    print(f"Passed            : {passed}/{total}")
    print(f"Failed            : {total - passed}/{total}")
    print("Regression status : " + ("PASS" if passed == total else "FAIL"))


if __name__ == "__main__":
    main()
