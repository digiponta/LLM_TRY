# eval_trained_concept_promotion_v1022.py
#
# v10.2.2 regression:
# all explicit concept query forms used by the pre-gate can be promoted after /train.

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from chat import (
    pair_fingerprint,
    pre_generation_unknown_concept,
    trained_known_concepts,
)


CASES = [
    ("宇宙とは", "宇宙"),
    ("数学は", "数学"),
    ("文学について教えて", "文学"),
    ("化学を説明して", "化学"),
    ("生物学を簡単に説明して", "生物学"),
    ("ブラックホールって何", "ブラックホール"),
]


def main() -> None:
    print("=" * 88)
    print(" LLM_TRY v10.2.2 Concept Query Promotion Regression")
    print("=" * 88)

    passed = 0
    total = 0

    with tempfile.TemporaryDirectory(prefix="llm_try_v1022_") as tmp:
        root = Path(tmp)
        log = root / "chat_history.jsonl"
        state = root / "chat_learning_state.json"

        rows = []
        fps = []
        for question, focus in CASES:
            answer = f"{focus}についての教示済み定義。"
            rows.append({
                "user": question,
                "assistant": answer,
                "source": "chat-manual",
            })
            fps.append(pair_fingerprint(question, answer))

        log.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        state.write_text(
            json.dumps(
                {"version": "v1.6.2", "trained_fingerprints": fps},
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        promoted = trained_known_concepts(log, state)

        for question, focus in CASES:
            total += 1
            unknown, extracted = pre_generation_unknown_concept(
                question,
                promoted_concepts=promoted,
            )
            ok = (not unknown and extracted == focus and focus.lower() in promoted)
            if ok:
                passed += 1
            print(
                f"[{'PASS' if ok else 'FAIL'}] {question} "
                f"-> focus={extracted}, promoted={focus.lower() in promoted}"
            )

        total += 1
        unknown, focus = pre_generation_unknown_concept(
            "未教示概念とは",
            promoted_concepts=promoted,
        )
        ok = unknown and focus == "未教示概念"
        if ok:
            passed += 1
        print(
            f"[{'PASS' if ok else 'FAIL'}] untrained remains unknown "
            f"-> focus={focus}"
        )

    print()
    print("Summary")
    print("-" * 88)
    print(f"Passed            : {passed}/{total}")
    print(f"Failed            : {total - passed}/{total}")
    print("Regression status : " + ("PASS" if passed == total else "FAIL"))


if __name__ == "__main__":
    main()
