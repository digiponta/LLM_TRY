# eval_persistent_checkpoint_v1024.py
#
# v10.2.4 regression:
# default startup resumes the online checkpoint only when training state proves
# at least one trusted pair has been consumed. Explicit --model always wins.

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from chat import DEFAULT_MODEL, choose_startup_model


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    suffix = f" : {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    return condition


def main() -> None:
    print("=" * 88)
    print(" LLM_TRY v10.2.4 Persistent Adaptive Checkpoint Regression")
    print("=" * 88)

    passed = 0
    total = 0

    def run(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, total
        total += 1
        if check(name, condition, detail):
            passed += 1

    with tempfile.TemporaryDirectory(prefix="llm_try_v1024_") as tmp:
        root = Path(tmp)
        base = root / "base.pt"
        online = root / "online.pt"
        explicit = root / "explicit.pt"
        state = root / "state.json"

        base.write_text("base", encoding="utf-8")
        explicit.write_text("explicit", encoding="utf-8")

        selected = choose_startup_model(
            str(base),
            online_output=str(online),
            learning_state=str(state),
        )
        run(
            "missing-state-keeps-base",
            selected == base,
            str(selected),
        )

        online.write_text("online", encoding="utf-8")
        state.write_text(
            json.dumps({"version": "v1.6.2", "trained_fingerprints": []}),
            encoding="utf-8",
        )
        selected = choose_startup_model(
            DEFAULT_MODEL,
            online_output=str(online),
            learning_state=str(state),
        )
        run(
            "empty-state-keeps-default-base",
            selected.name != online.name,
            str(selected),
        )

        state.write_text(
            json.dumps(
                {"version": "v1.6.2", "trained_fingerprints": ["abc"]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        selected = choose_startup_model(
            DEFAULT_MODEL,
            online_output=str(online),
            learning_state=str(state),
        )
        run(
            "trained-state-resumes-online",
            selected == online,
            str(selected),
        )

        selected = choose_startup_model(
            str(explicit),
            online_output=str(online),
            learning_state=str(state),
        )
        run(
            "explicit-model-wins",
            selected == explicit,
            str(selected),
        )

    print()
    print("Summary")
    print("-" * 88)
    print(f"Passed            : {passed}/{total}")
    print(f"Failed            : {total - passed}/{total}")
    print("Regression status : " + ("PASS" if passed == total else "FAIL"))


if __name__ == "__main__":
    main()
