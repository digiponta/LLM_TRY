from __future__ import annotations

from chat import parse_subject_fact, compose_context_fact_answer


def main() -> None:
    print("=" * 96)
    print(" LLM_TRY v10.7.2 Relation-Aware Fact Parsing Regression")
    print("=" * 96)

    passed = 0
    total = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, total
        total += 1
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} : {detail}")

    cases = [
        (
            "includes",
            "文学は、分類上、数学を含む",
            {
                "subject": "文学",
                "relation": "includes",
                "value": "数学",
                "relation_context": "分類上",
            },
        ),
        (
            "belongs-to",
            "東京は日本に属する",
            {
                "subject": "東京",
                "relation": "belongs_to",
                "value": "日本",
            },
        ),
        (
            "has",
            "鳥は翼を持つ",
            {
                "subject": "鳥",
                "relation": "has",
                "value": "翼",
            },
        ),
        (
            "used-for",
            "GPUは並列計算に利用される",
            {
                "subject": "GPU",
                "relation": "used_for",
                "value": "並列計算",
            },
        ),
    ]

    parsed_rows = []
    for name, text, expected in cases:
        row = parse_subject_fact(text)
        parsed_rows.append(row)
        check(f"parse-{name}", row is not None, repr(row))
        if row is not None:
            for key, value in expected.items():
                check(
                    f"{name}-{key}",
                    row.get(key) == value,
                    f"{row.get(key)!r}",
                )

    literature = parsed_rows[0]
    if literature is not None:
        rendered = compose_context_fact_answer("文学", [literature])
        check(
            "render-includes",
            rendered == "文学は、分類上、数学を含む。",
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
