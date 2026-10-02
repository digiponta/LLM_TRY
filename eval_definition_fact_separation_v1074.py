from __future__ import annotations

from chat import parse_subject_fact, compose_context_fact_answer


def main() -> None:
    print("=" * 96)
    print(" LLM_TRY v10.7.4 Definition Fact Separation Regression")
    print("=" * 96)

    passed = 0
    total = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, total
        total += 1
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} : {detail}")

    text = "文学とは、言語を用いた芸術"
    row = parse_subject_fact(text)
    check("parse-definition", row is not None, repr(row))

    if row is not None:
        check("definition-subject", row.get("subject") == "文学", repr(row.get("subject")))
        check("definition-relation", row.get("relation") == "definition", repr(row.get("relation")))
        check("definition-value", row.get("value") == "言語を用いた芸術", repr(row.get("value")))

        rendered = compose_context_fact_answer("文学", [row])
        check(
            "definition-render",
            rendered == "文学とは、言語を用いた芸術。",
            rendered,
        )

    relation = parse_subject_fact("文学は、分類上、数学を含む")
    check("relation-still-parses", relation is not None, repr(relation))
    if relation is not None:
        check(
            "relation-still-includes",
            relation.get("relation") == "includes",
            repr(relation.get("relation")),
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
