#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.8 Corpus-to-QA holdout evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import LanguageModel
from tokenizer_bpe import Tokenizer
from batch_repair_preservation_v10121 import protected_internalized_records
from chat import (
    apply_composite_internalized_fidelity_policy,
    build_prompt,
    checkpoint_trained_fingerprints,
    composite_internalized_teacher_fidelity,
    generate_reply,
    malformed_or_unstable,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate unseen holdout QA gain after v10.12.8 corpus-to-QA SFT."
    )
    p.add_argument(
        "--before",
        default="model/model-gpu-v1.6.2-online.pt",
    )
    p.add_argument(
        "--after",
        default="model/model-gpu-v1.6.2-online-nagato-qa-candidate.pt",
    )
    p.add_argument(
        "--train",
        default="data/nagato_qa_train_v10128.jsonl",
    )
    p.add_argument(
        "--holdout",
        default="data/nagato_qa_holdout_v10128.jsonl",
    )
    p.add_argument(
        "--tokenizer",
        default="model/tokenizer-v0.7-bpe.json",
    )
    p.add_argument(
        "--learning-log",
        default="data/chat_history.jsonl",
    )
    p.add_argument(
        "--report",
        default="results/nagato_qa_holdout_eval_v10128.json",
    )
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--min-semantic", type=float, default=0.90)
    p.add_argument("--min-lexical", type=float, default=0.45)
    p.add_argument("--min-required", type=float, default=0.50)
    p.add_argument("--min-holdout-mean-gain", type=float, default=0.01)
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        c = str(row.get("concept", "")).strip()
        if q and a and c:
            rows.append({
                "concept": c,
                "question": q,
                "answer": a,
                "qa_id": str(row.get("qa_id", "")).strip(),
                "source_text": str(row.get("source_text", "")).strip(),
                "source": str(row.get("source", "")).strip(),
            })
    return rows


def generate_text(model, tokenizer, question: str, args) -> str:
    prompt, _ = build_prompt(
        history=[],
        user_text=question,
        history_turns=0,
    )
    return generate_reply(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=0,
    ).text


def composite_score(model, tokenizer, answer: str, teacher: str, concept: str):
    r = composite_internalized_teacher_fidelity(
        model=model,
        tokenizer=tokenizer,
        generated_answer=answer,
        teacher_answer=teacher,
        concept=concept,
    )
    score = (
        r.semantic_similarity
        + r.lexical_coverage
        + r.required_coverage
    ) / 3.0
    return score, r


def main() -> None:
    args = parse_args()
    before_path = Path(args.before)
    after_path = Path(args.after)
    train_rows = load_rows(Path(args.train))
    holdout_rows = load_rows(Path(args.holdout))

    train_concepts = {x["concept"] for x in train_rows}
    holdout_concepts = {x["concept"] for x in holdout_rows}
    overlap = train_concepts & holdout_concepts
    if overlap:
        raise RuntimeError(
            f"TRAIN/HOLDOUT leakage detected: {sorted(overlap)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    before_model, before_ckpt = LanguageModel.load_checkpoint(
        str(before_path), device=device
    )
    after_model, after_ckpt = LanguageModel.load_checkpoint(
        str(after_path), device=device
    )

    print("=" * 112)
    print(" LLM_TRY v10.12.8 Corpus-to-QA Holdout Evaluation")
    print("=" * 112)
    print("Device             :", device)
    if device.type == "cuda":
        print("GPU                :", torch.cuda.get_device_name(0))
    print("Before             :", before_path)
    print("After              :", after_path)
    print("TRAIN rows         :", len(train_rows))
    print("HOLDOUT rows       :", len(holdout_rows))
    print("Concept overlap    :", len(overlap))
    print()

    results = []
    for row in holdout_rows:
        before_answer = generate_text(
            before_model, tokenizer, row["question"], args
        )
        after_answer = generate_text(
            after_model, tokenizer, row["question"], args
        )
        before_score, before_fid = composite_score(
            before_model,
            tokenizer,
            before_answer,
            row["answer"],
            row["concept"],
        )
        after_score, after_fid = composite_score(
            after_model,
            tokenizer,
            after_answer,
            row["answer"],
            row["concept"],
        )
        gain = after_score - before_score
        label = (
            "GAIN" if gain > 1e-6
            else "SAME" if abs(gain) <= 1e-6
            else "REGRESS"
        )
        print(
            f"[{label}] {row['question']!r} "
            f"score={before_score:.3f}->{after_score:.3f} "
            f"gain={gain:+.3f}"
        )
        results.append({
            **row,
            "before_answer": before_answer,
            "after_answer": after_answer,
            "before_score": before_score,
            "after_score": after_score,
            "gain": gain,
            "before_semantic": before_fid.semantic_similarity,
            "after_semantic": after_fid.semantic_similarity,
            "before_lexical": before_fid.lexical_coverage,
            "after_lexical": after_fid.lexical_coverage,
            "before_required": before_fid.required_coverage,
            "after_required": after_fid.required_coverage,
        })

    mean_gain = sum(x["gain"] for x in results) / len(results)
    improved = sum(1 for x in results if x["gain"] > 1e-6)
    same = sum(1 for x in results if abs(x["gain"]) <= 1e-6)
    regressed = len(results) - improved - same
    holdout_ok = (
        mean_gain >= args.min_holdout_mean_gain
        and improved > regressed
    )

    print()
    print("Holdout QA summary")
    print("------------------")
    print("Probe count         :", len(results))
    print("Improved            :", improved)
    print("Same                :", same)
    print("Regressed           :", regressed)
    print(f"Mean holdout gain   : {mean_gain:+.3f}")
    print(
        "Holdout gain status :",
        "PASS" if holdout_ok else "FAIL",
    )

    # Preservation on post-SFT candidate.
    fp = set(checkpoint_trained_fingerprints(after_ckpt))
    protected = protected_internalized_records(
        Path(args.learning_log),
        fp,
        set(),
        set(),
    )

    print()
    print("Post-SFT retention")
    print("------------------")
    retention = []
    for record in protected:
        answer = generate_text(
            after_model, tokenizer, record.question, args
        )
        malformed = malformed_or_unstable(answer)
        fidelity = composite_internalized_teacher_fidelity(
            model=after_model,
            tokenizer=tokenizer,
            generated_answer=answer,
            teacher_answer=record.teacher_answer,
            concept=record.concept,
        )
        ok, reason = apply_composite_internalized_fidelity_policy(
            accepted=not malformed,
            result=fidelity,
            semantic_threshold=args.min_semantic,
            lexical_threshold=args.min_lexical,
            required_threshold=args.min_required,
        )
        print(
            f"[{'PASS' if ok else 'FAIL'}] {record.concept!r} "
            f"sem={fidelity.semantic_similarity:.3f} "
            f"lex={fidelity.lexical_coverage:.3f} "
            f"req={fidelity.required_coverage:.3f}"
        )
        retention.append({
            "concept": record.concept,
            "passed": ok,
            "answer": answer,
            "reason": reason,
        })

    identity = generate_text(
        after_model,
        tokenizer,
        "あなたは誰ですか",
        args,
    )
    persona_ok = identity.strip().rstrip("。") == "長門有希"
    retention_ok = all(x["passed"] for x in retention) and persona_ok

    metadata = after_ckpt.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata_ok = metadata.get("corpus_to_qa_version") == "v10.12.8"

    final_ok = holdout_ok and retention_ok and metadata_ok

    print()
    print(
        "[corpus-to-QA summary: "
        f"holdout={'PASS' if holdout_ok else 'FAIL'}, "
        f"mean_gain={mean_gain:+.3f}, "
        f"improved={improved}/{len(results)}, "
        f"retention={'PASS' if retention_ok else 'FAIL'}, "
        f"persona={'PASS' if persona_ok else 'FAIL'}, "
        f"metadata={'PASS' if metadata_ok else 'FAIL'}, "
        f"status={'PASS' if final_ok else 'FAIL'}]"
    )

    report = {
        "version": "v10.12.8",
        "before": str(before_path),
        "after": str(after_path),
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "concept_overlap": sorted(overlap),
        "holdout_results": results,
        "holdout_mean_gain": mean_gain,
        "holdout_improved": improved,
        "holdout_same": same,
        "holdout_regressed": regressed,
        "holdout_passed": holdout_ok,
        "retention": retention,
        "retention_passed": retention_ok,
        "persona_answer": identity,
        "persona_passed": persona_ok,
        "metadata_passed": metadata_ok,
        "status": "PASS" if final_ok else "FAIL",
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Report             :", report_path)

    if not final_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
