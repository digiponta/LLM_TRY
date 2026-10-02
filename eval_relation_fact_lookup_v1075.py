from __future__ import annotations

import json
from pathlib import Path
import tempfile

from chat import (
    append_fact_store,
    compose_context_fact_answer,
    pair_fingerprint,
    parse_subject_fact,
    trained_facts,
)


def main() -> None:
    print("=" * 96)
    print(" LLM_TRY v10.7.5 Relation Fact Lookup Regression")
    print("=" * 96)

    passed = 0
    total = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, total
        total += 1
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} : {detail}")

    with tempfile.TemporaryDirectory(prefix="llm_try_v1075_") as tmp:
        root = Path(tmp)
        store = root / "fact_store.jsonl"
        state = root / "state.json"

        q = "文学は、数学を含む"
        a = "文学は、数学を含む"

        check("save-relation-fact", append_fact_store(store, q, a), a)

        state.write_text(
            json.dumps(
                {
                    "version": "v1.6.2",
                    "trained_fingerprints": [pair_fingerprint(q, a)],
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        query = parse_subject_fact(q)
        check("parse-relation-query", query is not None, repr(query))

        facts = trained_facts(store, state, "文学")
        check("trained-relation-present", len(facts) == 1, repr(facts))

        if query is not None:
            matches = [
                fact for fact in facts
                if fact.get("relation") == query.get("relation")
                and fact.get("value") == query.get("value")
                and str(fact.get("condition", "")) == str(query.get("condition", ""))
                and str(fact.get("relation_context", "")) == str(query.get("relation_context", ""))
            ]
            check("exact-relation-match", len(matches) == 1, repr(matches))

            if matches:
                rendered = compose_context_fact_answer("文学", [matches[0]])
                check(
                    "relation-render",
                    rendered == "文学は、数学を含む。",
                    rendered,
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
