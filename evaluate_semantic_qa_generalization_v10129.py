#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.9 Semantic QA holdout generalization evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import LanguageModel
from tokenizer_bpe import Tokenizer
from semantic_qa_generalization_v10129 import decompose_definition, semantic_prompt
from batch_repair_preservation_v10121 import protected_internalized_records
from chat import (
    apply_composite_internalized_fidelity_policy,
    build_prompt,
    checkpoint_trained_fingerprints,
    composite_internalized_teacher_fidelity,
    generate_reply,
    malformed_or_unstable,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Measure plain vs semantic-query generalization on unseen HOLDOUT concepts."
    )
    p.add_argument("--before", default="model/model-gpu-v1.6.2-online.pt")
    p.add_argument(
        "--after",
        default="model/model-gpu-v1.6.2-online-semantic-qa-candidate.pt",
    )
    p.add_argument("--train", default="data/nagato_qa_train_v10128.jsonl")
    p.add_argument("--holdout", default="data/nagato_qa_holdout_v10128.jsonl")
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    p.add_argument("--learning-log", default="data/chat_history.jsonl")
    p.add_argument(
        "--report",
        default="results/semantic_qa_generalization_v10129.json",
    )
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--min-semantic", type=float, default=0.90)
    p.add_argument("--min-lexical", type=float, default=0.45)
    p.add_argument("--min-required", type=float, default=0.50)
    p.add_argument("--min-mean-gain", type=float, default=0.01)
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        concept = str(row.get("concept", "")).strip()
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if concept and question and answer:
            rows.append({
                "concept": concept,
                "question": question,
                "answer": answer,
                "source_text": str(row.get("source_text", "")).strip(),
            })
    return rows


def generate(model, tokenizer, query: str, args) -> str:
    prompt, _ = build_prompt(
        history=[],
        user_text=query,
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


def score(model, tokenizer, answer, teacher, concept) -> float:
    r = composite_internalized_teacher_fidelity(
        model=model,
        tokenizer=tokenizer,
        generated_answer=answer,
        teacher_answer=teacher,
        concept=concept,
    )
    return (
        r.semantic_similarity
        + r.lexical_coverage
        + r.required_coverage
    ) / 3.0


def main() -> None:
    args = parse_args()
    train_rows = load_rows(Path(args.train))
    holdout_rows = load_rows(Path(args.holdout))

    train_concepts = {r["concept"] for r in train_rows}
    holdout_concepts = {r["concept"] for r in holdout_rows}
    overlap = train_concepts & holdout_concepts
    if overlap:
        raise RuntimeError(f"TRAIN/HOLDOUT leakage: {sorted(overlap)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    before_model, before_ckpt = LanguageModel.load_checkpoint(
        args.before, device=device
    )
    after_model, after_ckpt = LanguageModel.load_checkpoint(
        args.after, device=device
    )

    print("=" * 116)
    print(" LLM_TRY v10.12.9 Semantic QA Generalization Evaluation")
    print("=" * 116)
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Before              :", args.before)
    print("After               :", args.after)
    print("TRAIN concepts      :", len(train_concepts))
    print("HOLDOUT concepts    :", len(holdout_concepts))
    print("Concept overlap     :", len(overlap))
    print()

    results = []
    for row in holdout_rows:
        item = decompose_definition(
            row["concept"], row["question"], row["answer"]
        )
        semantic_query = semantic_prompt(item)

        before_plain_answer = generate(
            before_model, tokenizer, item.question, args
        )
        after_plain_answer = generate(
            after_model, tokenizer, item.question, args
        )
        before_sem_answer = generate(
            before_model, tokenizer, semantic_query, args
        )
        after_sem_answer = generate(
            after_model, tokenizer, semantic_query, args
        )

        before_plain = score(
            before_model, tokenizer, before_plain_answer, item.answer, item.concept
        )
        after_plain = score(
            after_model, tokenizer, after_plain_answer, item.answer, item.concept
        )
        before_sem = score(
            before_model, tokenizer, before_sem_answer, item.answer, item.concept
        )
        after_sem = score(
            after_model, tokenizer, after_sem_answer, item.answer, item.concept
        )

        plain_gain = after_plain - before_plain
        semantic_gain = after_sem - before_sem
        semantic_lift = after_sem - after_plain

        label = (
            "GAIN" if semantic_gain > 1e-6
            else "SAME" if abs(semantic_gain) <= 1e-6
            else "REGRESS"
        )
        print(
            f"[{label}] {item.concept!r} "
            f"plain={before_plain:.3f}->{after_plain:.3f} "
            f"semantic={before_sem:.3f}->{after_sem:.3f} "
            f"sem_gain={semantic_gain:+.3f} "
            f"sem_lift={semantic_lift:+.3f}"
        )

        results.append({
            "concept": item.concept,
            "question": item.question,
            "semantic_query": semantic_query,
            "teacher": item.answer,
            "before_plain_answer": before_plain_answer,
            "after_plain_answer": after_plain_answer,
            "before_semantic_answer": before_sem_answer,
            "after_semantic_answer": after_sem_answer,
            "before_plain_score": before_plain,
            "after_plain_score": after_plain,
            "before_semantic_score": before_sem,
            "after_semantic_score": after_sem,
            "plain_gain": plain_gain,
            "semantic_gain": semantic_gain,
            "semantic_lift": semantic_lift,
        })

    mean_plain_gain = sum(x["plain_gain"] for x in results) / len(results)
    mean_semantic_gain = sum(x["semantic_gain"] for x in results) / len(results)
    mean_semantic_lift = sum(x["semantic_lift"] for x in results) / len(results)
    improved = sum(1 for x in results if x["semantic_gain"] > 1e-6)
    same = sum(1 for x in results if abs(x["semantic_gain"]) <= 1e-6)
    regressed = len(results) - improved - same

    generalization_ok = (
        mean_semantic_gain >= args.min_mean_gain
        and improved > regressed
    )

    print()
    print("Semantic generalization summary")
    print("-------------------------------")
    print(f"Mean plain gain      : {mean_plain_gain:+.3f}")
    print(f"Mean semantic gain   : {mean_semantic_gain:+.3f}")
    print(f"Mean semantic lift   : {mean_semantic_lift:+.3f}")
    print("Improved             :", improved)
    print("Same                 :", same)
    print("Regressed            :", regressed)
    print(
        "Generalization status:",
        "PASS" if generalization_ok else "FAIL",
    )

    fp = set(checkpoint_trained_fingerprints(after_ckpt))
    protected = protected_internalized_records(
        Path(args.learning_log),
        fp,
        set(),
        set(),
    )

    print()
    print("Post-semantic-SFT retention")
    print("---------------------------")
    retention = []
    for record in protected:
        answer = generate(after_model, tokenizer, record.question, args)
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
            "reason": reason,
        })

    identity = generate(after_model, tokenizer, "あなたは誰ですか", args)
    persona_ok = identity.strip().rstrip("。") == "長門有希"
    retention_ok = all(x["passed"] for x in retention) and persona_ok

    metadata = after_ckpt.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata_ok = metadata.get("semantic_qa_version") == "v10.12.9"

    final_ok = generalization_ok and retention_ok and metadata_ok

    print()
    print(
        "[semantic QA summary: "
        f"generalization={'PASS' if generalization_ok else 'FAIL'}, "
        f"mean_sem_gain={mean_semantic_gain:+.3f}, "
        f"mean_sem_lift={mean_semantic_lift:+.3f}, "
        f"improved={improved}/{len(results)}, "
        f"retention={'PASS' if retention_ok else 'FAIL'}, "
        f"persona={'PASS' if persona_ok else 'FAIL'}, "
        f"metadata={'PASS' if metadata_ok else 'FAIL'}, "
        f"status={'PASS' if final_ok else 'FAIL'}]"
    )

    report = {
        "version": "v10.12.9",
        "before": args.before,
        "after": args.after,
        "train_concepts": len(train_concepts),
        "holdout_concepts": len(holdout_concepts),
        "concept_overlap": sorted(overlap),
        "results": results,
        "mean_plain_gain": mean_plain_gain,
        "mean_semantic_gain": mean_semantic_gain,
        "mean_semantic_lift": mean_semantic_lift,
        "improved": improved,
        "same": same,
        "regressed": regressed,
        "generalization_passed": generalization_ok,
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
    print("Report              :", report_path)

    if not final_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
