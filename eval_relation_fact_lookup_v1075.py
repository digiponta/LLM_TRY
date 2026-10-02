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
    print(" LLM_TRY v10.7.6 Context-Tolerant Relation Lookup Regression")
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

        q2 = "文学は、分類上、数学を含む"
        a2 = "文学は、分類上、数学を含む"
        check("save-contextual-relation", append_fact_store(store, q2, a2), a2)

        state.write_text(
            json.dumps(
                {
                    "version": "v1.6.2",
                    "trained_fingerprints": [
                        pair_fingerprint(q, a),
                        pair_fingerprint(q2, a2),
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        query_no_context = parse_subject_fact("文学は、数学を含む")
        facts = trained_facts(store, state, "文学")
        if query_no_context is not None:
            qc = str(query_no_context.get("relation_context", "")).strip()
            qcond = str(query_no_context.get("condition", "")).strip()
            tolerant = [
                fact for fact in facts
                if fact.get("relation") == query_no_context.get("relation")
                and fact.get("value") == query_no_context.get("value")
                and (not qcond or str(fact.get("condition", "")).strip() == qcond)
                and (not qc or str(fact.get("relation_context", "")).strip() == qc)
            ]
            check(
                "context-omitted-query-matches",
                len(tolerant) >= 1,
                repr(tolerant),
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
