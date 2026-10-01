# audit_nagato_training_conflicts.py
#
# Audit all Nagato SFT datasets for conflicting supervision.
#
# Detects:
#   1. Exact same user prompt with multiple different answers
#   2. Normalized prompt collisions
#   3. "未学習です" accidentally used as a training target
#   4. Very short vs much longer answers for the same normalized prompt
#   5. Near-duplicate prompts with strongly different answers
#
# This is intended for LLM_TRY v9 before further SFT.

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_FILES = [
    "data/nagato_chat.jsonl",
    "data/nagato_identity_anchor.jsonl",
    "data/nagato_response_expansion.jsonl",
    "data/nagato_response_completion.jsonl",
    "data/nagato_paraphrase_generalization.jsonl",
    "data/nagato_semantic_consistency.jsonl",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit Nagato SFT training conflicts.")
    p.add_argument(
        "files",
        nargs="*",
        default=DEFAULT_FILES,
        help="JSONL datasets to audit.",
    )
    p.add_argument(
        "--near-prompt-threshold",
        type=float,
        default=0.82,
        help="Similarity threshold for near-duplicate user prompts.",
    )
    p.add_argument(
        "--answer-sim-threshold",
        type=float,
        default=0.45,
        help="If answer similarity is below this, flag as possible conflict.",
    )
    p.add_argument(
        "--length-ratio-threshold",
        type=float,
        default=2.5,
        help="Flag same-prompt answers when longest/shortest exceeds this ratio.",
    )
    return p.parse_args()


def normalize_text(text: str) -> str:
    text = text.strip()
    text = text.replace("？", "?")
    text = re.sub(r"[\s　]+", "", text)
    text = re.sub(r"[。.!！?？]+$", "", text)
    return text.lower()


def read_rows(path: Path) -> List[dict]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        user = str(row.get("user", "")).strip()
        assistant = str(row.get("assistant", "")).strip()
        if not user or not assistant:
            continue
        rows.append({
            "file": str(path),
            "line": line_no,
            "user": user,
            "assistant": assistant,
            "norm_user": normalize_text(user),
        })
    return rows


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def answer_len(text: str) -> int:
    return max(1, len(normalize_text(text)))


def print_entries(entries: List[dict]) -> None:
    for e in entries:
        print(f"  - {e['file']}:{e['line']}")
        print(f"    Q: {e['user']}")
        print(f"    A: {e['assistant']}")


def main() -> None:
    args = parse_args()

    rows: List[dict] = []
    for filename in args.files:
        path = Path(filename)
        if not path.exists():
            print(f"[WARN] missing: {path}")
            continue
        rows.extend(read_rows(path))

    print("=" * 88)
    print(" LLM_TRY Nagato SFT Training Conflict Audit")
    print("=" * 88)
    print("Datasets:")
    for filename in args.files:
        print(" ", filename)
    print("Rows:", len(rows))
    print()

    exact: Dict[str, List[dict]] = defaultdict(list)
    normalized: Dict[str, List[dict]] = defaultdict(list)

    for row in rows:
        exact[row["user"]].append(row)
        normalized[row["norm_user"]].append(row)

    exact_conflicts = []
    for q, entries in exact.items():
        answers = {e["assistant"] for e in entries}
        if len(answers) > 1:
            exact_conflicts.append((q, entries))

    print("A) Exact prompt / multiple answers")
    print("-" * 88)
    if not exact_conflicts:
        print("PASS: no exact-prompt conflicts")
    else:
        for q, entries in exact_conflicts:
            print(f"[CONFLICT] {q}")
            print_entries(entries)
            print()

    print()
    print("B) Normalized prompt collisions")
    print("-" * 88)
    norm_conflicts = []
    for q, entries in normalized.items():
        answers = {normalize_text(e["assistant"]) for e in entries}
        if len(entries) > 1 and len(answers) > 1:
            norm_conflicts.append((q, entries))

    if not norm_conflicts:
        print("PASS: no normalized-prompt conflicts")
    else:
        for q, entries in norm_conflicts:
            print(f"[CONFLICT] normalized={q}")
            print_entries(entries)
            print()

    print()
    print('C) "未学習です" used as training target')
    print("-" * 88)
    unknown_targets = [r for r in rows if "未学習です" in r["assistant"]]
    if not unknown_targets:
        print("PASS: no unknown fallback used as target")
    else:
        print_entries(unknown_targets)

    print()
    print("D) Short-vs-long answer collisions")
    print("-" * 88)
    length_conflicts = []
    for q, entries in normalized.items():
        if len(entries) < 2:
            continue
        lengths = [answer_len(e["assistant"]) for e in entries]
        mn = min(lengths)
        mx = max(lengths)
        if mx / mn >= args.length_ratio_threshold:
            length_conflicts.append((q, entries, mn, mx, mx / mn))

    if not length_conflicts:
        print("PASS: no severe answer-length collisions")
    else:
        for q, entries, mn, mx, ratio in length_conflicts:
            print(f"[LENGTH] normalized={q} min={mn} max={mx} ratio={ratio:.2f}")
            print_entries(entries)
            print()

    print()
    print("E) Near-duplicate prompts with divergent answers")
    print("-" * 88)

    # Deduplicate identical normalized prompts here; section B already covers them.
    unique_prompts: Dict[str, dict] = {}
    for row in rows:
        unique_prompts.setdefault(row["norm_user"], row)
    items = list(unique_prompts.values())

    near_conflicts: List[Tuple[float, float, dict, dict]] = []
    for i in range(len(items)):
        a = items[i]
        for j in range(i + 1, len(items)):
            b = items[j]
            ps = sim(a["user"], b["user"])
            if ps < args.near_prompt_threshold:
                continue
            ans = sim(a["assistant"], b["assistant"])
            if ans < args.answer_sim_threshold:
                near_conflicts.append((ps, ans, a, b))

    near_conflicts.sort(key=lambda x: (-x[0], x[1]))

    if not near_conflicts:
        print("PASS: no suspicious near-duplicate conflicts")
    else:
        for ps, ans, a, b in near_conflicts:
            print(f"[NEAR] prompt_sim={ps:.3f} answer_sim={ans:.3f}")
            print(f"  A {a['file']}:{a['line']}")
            print(f"    Q: {a['user']}")
            print(f"    A: {a['assistant']}")
            print(f"  B {b['file']}:{b['line']}")
            print(f"    Q: {b['user']}")
            print(f"    A: {b['assistant']}")
            print()

    print()
    print("Summary")
    print("-" * 88)
    print("Exact conflicts       :", len(exact_conflicts))
    print("Normalized conflicts  :", len(norm_conflicts))
    print("Unknown targets       :", len(unknown_targets))
    print("Length conflicts      :", len(length_conflicts))
    print("Near-prompt conflicts :", len(near_conflicts))


if __name__ == "__main__":
    main()
