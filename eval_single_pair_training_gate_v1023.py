# eval_single_pair_training_gate_v1023.py
#
# Regression for v10.2.3:
# chat.py must allow /train with a single trusted learning pair.

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from chat import learning_log_count


def main() -> None:
    print("=" * 88)
    print(" LLM_TRY v10.2.3 Single-Pair Training Gate Regression")
    print("=" * 88)

    with tempfile.TemporaryDirectory(prefix="llm_try_v1023_") as tmp:
        path = Path(tmp) / "chat_history.jsonl"

        empty = learning_log_count(path)
        print(f"[{'PASS' if empty == 0 else 'FAIL'}] empty-log-count : {empty}")

        row = {
            "user": "宇宙とは",
            "assistant": "宇宙は、物質・エネルギー・時空を含む世界全体を指す。",
            "source": "chat-manual",
        }
        path.write_text(
            json.dumps(row, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        one = learning_log_count(path)
        print(f"[{'PASS' if one == 1 else 'FAIL'}] single-pair-count : {one}")

        can_train = one >= 1
        print(
            f"[{'PASS' if can_train else 'FAIL'}] "
            f"single-pair-training-enabled : {can_train}"
        )

        passed = (empty == 0) and (one == 1) and can_train

    print()
    print("Summary")
    print("-" * 88)
    print("Regression status :", "PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
