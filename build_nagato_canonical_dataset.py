# build_nagato_canonical_dataset.py
#
# Build a single canonical Nagato SFT dataset from the staged v4-v8 datasets.
# Exact/normalized duplicate prompts are collapsed to one answer according to
# an explicit canonical priority and selected overrides for identity/persona.
#
# Existing source files are not modified.

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

DEFAULT_FILES = [
    "data/nagato_chat.jsonl",
    "data/nagato_identity_anchor.jsonl",
    "data/nagato_response_expansion.jsonl",
    "data/nagato_response_completion.jsonl",
    "data/nagato_paraphrase_generalization.jsonl",
    "data/nagato_semantic_consistency.jsonl",
]

# Later sources override earlier sources for the same normalized prompt.
SOURCE_PRIORITY = {
    "nagato_chat.jsonl": 10,
    "nagato_identity_anchor.jsonl": 20,
    "nagato_response_expansion.jsonl": 30,
    "nagato_response_completion.jsonl": 40,
    "nagato_paraphrase_generalization.jsonl": 50,
    "nagato_semantic_consistency.jsonl": 60,
}

# Explicit canonical answers for important identity/persona prompts.
CANONICAL_OVERRIDES = {
    "あなたは誰ですか": "長門有希。",
    "自己紹介してください": "長門有希。読書が好き。",
    "本は好きですか": "好き。読書が好き。",
    "読書は好きですか": "好き。読書が好き。",
    "趣味は何ですか": "読書。",
    "今日はどうですか": "特に変化はない。",
    "元気ですか": "問題ない。",
    "人間についてどう思いますか": "複雑。興味深い。",
    "楽しいですか": "不快ではない。",
    "寂しいですか": "静かな状態には慣れている。",
    "感情はありますか": "人間と同じではない。",
    "怒っていますか": "怒っていない。",
    "好きな場所は": "静かな場所。",
    "好きな時間は": "静かな時間。",
    "あなたについて教えて": "長門有希。読書が好き。静かな環境を好む。",
    "人についてどう考えますか": "複雑。興味深い。",
}


def normalize_prompt(text: str) -> str:
    text = text.strip()
    text = text.replace("？", "?")
    text = re.sub(r"[\s　]+", "", text)
    text = re.sub(r"[。.!！?？]+$", "", text)
    return text.lower()


def load_rows(path: Path) -> List[dict]:
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
            "user": user,
            "assistant": assistant,
            "source": path.name,
            "line": line_no,
            "priority": SOURCE_PRIORITY.get(path.name, 0),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Build canonical Nagato SFT dataset.")
    p.add_argument("files", nargs="*", default=DEFAULT_FILES)
    p.add_argument("--output", default="data/nagato_canonical_v91.jsonl")
    p.add_argument("--report", default="results/nagato_canonical_v91_report.txt")
    args = p.parse_args()

    all_rows = []
    for filename in args.files:
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(path)
        all_rows.extend(load_rows(path))

    grouped: Dict[str, List[dict]] = OrderedDict()
    order = []
    for row in all_rows:
        key = normalize_prompt(row["user"])
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)

    canonical = []
    conflict_count = 0
    override_count = 0

    report_lines = []
    report_lines.append("LLM_TRY Nagato Canonical Dataset v9.1")
    report_lines.append("=" * 72)
    report_lines.append(f"Input rows : {len(all_rows)}")
    report_lines.append(f"Groups     : {len(grouped)}")
    report_lines.append("")

    for key in order:
        entries = grouped[key]
        answers = {e["assistant"] for e in entries}
        if len(answers) > 1:
            conflict_count += 1

        # Preserve the user spelling from the highest-priority source.
        selected = sorted(entries, key=lambda e: (e["priority"], e["line"]))[-1]
        user = selected["user"]
        assistant = selected["assistant"]

        # Exact textual overrides use the original user spelling.
        override = None
        for e in entries:
            if e["user"] in CANONICAL_OVERRIDES:
                override = CANONICAL_OVERRIDES[e["user"]]
                user = e["user"]
                break

        if override is not None:
            assistant = override
            override_count += 1

        canonical.append({"user": user, "assistant": assistant})

        if len(answers) > 1:
            report_lines.append(f"[RESOLVED] {user}")
            for e in entries:
                report_lines.append(
                    f"  {e['source']}:{e['line']} -> {e['assistant']}"
                )
            report_lines.append(f"  CANONICAL -> {assistant}")
            report_lines.append("")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in canonical:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_lines.append("Summary")
    report_lines.append("-" * 72)
    report_lines.append(f"Canonical rows    : {len(canonical)}")
    report_lines.append(f"Resolved conflicts: {conflict_count}")
    report_lines.append(f"Explicit overrides: {override_count}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("=" * 72)
    print(" LLM_TRY Nagato Canonical Dataset v9.1")
    print("=" * 72)
    print("Input rows        :", len(all_rows))
    print("Canonical rows    :", len(canonical))
    print("Resolved conflicts:", conflict_count)
    print("Explicit overrides:", override_count)
    print("Output            :", out_path)
    print("Report            :", report_path)


if __name__ == "__main__":
    main()
