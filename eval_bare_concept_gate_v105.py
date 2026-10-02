# eval_bare_concept_gate_v105.py
#
# LLM_TRY v10.5 regression:
# bare unknown concepts are rejected before generation, while static known and
# trained/promoted bare concepts remain available.

from __future__ import annotations

from chat import pre_generation_unknown_concept


STATIC_KNOWN = [
    "AI",
    "LLM",
    "CUDA",
    "GPU",
    "量子力学",
]

PROMOTED = {
    "宇宙",
    "数学",
    "文学",
}

BARE_UNKNOWN = [
    "ブラックホール",
    "相対性理論",
    "化学",
    "生物学",
    "歴史",
]


def main() -> None:
    print("=" * 88)
    print(" LLM_TRY v10.5 Bare Concept Gate Regression")
    print("=" * 88)

    passed = 0
    total = 0

    def run(name: str, ok: bool, detail: str) -> None:
        nonlocal passed, total
        total += 1
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} : {detail}")

    for q in STATIC_KNOWN:
        unknown, focus = pre_generation_unknown_concept(
            q,
            promoted_concepts={x.lower() for x in PROMOTED},
        )
        run(
            f"static-known {q}",
            not unknown and focus == q,
            f"unknown={unknown}, focus={focus}",
        )

    for q in sorted(PROMOTED):
        unknown, focus = pre_generation_unknown_concept(
            q,
            promoted_concepts={x.lower() for x in PROMOTED},
        )
        run(
            f"promoted-known {q}",
            not unknown and focus == q,
            f"unknown={unknown}, focus={focus}",
        )

    for q in BARE_UNKNOWN:
        unknown, focus = pre_generation_unknown_concept(
            q,
            promoted_concepts={x.lower() for x in PROMOTED},
        )
        run(
            f"bare-unknown {q}",
            unknown and focus == q,
            f"unknown={unknown}, focus={focus}",
        )

    print()
    print("Summary")
    print("-" * 88)
    print(f"Passed            : {passed}/{total}")
    print(f"Failed            : {total - passed}/{total}")
    print("Regression status : " + ("PASS" if passed == total else "FAIL"))

    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
