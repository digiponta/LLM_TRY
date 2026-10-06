#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.10 Semantic Role Generalization evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import LanguageModel
from tokenizer_bpe import Tokenizer
from semantic_role_generalization_v101210 import RoleProposition, semantic_role_prompt
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
        description="Evaluate seen-role/unseen-concept and unseen-role generalization."
    )
    p.add_argument("--before", default="model/model-gpu-v1.6.2-online.pt")
    p.add_argument(
        "--after",
        default="model/model-gpu-v1.6.2-online-semantic-role-candidate.pt",
    )
    p.add_argument("--seen-holdout", default="data/nagato_role_holdout_seen_v101210.jsonl")
    p.add_argument("--unseen-holdout", default="data/nagato_role_holdout_unseen_v101210.jsonl")
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    p.add_argument("--learning-log", default="data/chat_history.jsonl")
    p.add_argument("--report", default="results/semantic_role_generalization_v101210.json")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--min-semantic", type=float, default=0.90)
    p.add_argument("--min-lexical", type=float, default=0.45)
    p.add_argument("--min-required", type=float, default=0.50)
    p.add_argument("--min-seen-role-gain", type=float, default=0.01)
    p.add_argument("--min-unseen-role-gain", type=float, default=0.00)
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        rows.append(row)
    return rows


def generate(model, tokenizer, query: str, args) -> str:
    prompt, _ = build_prompt(history=[], user_text=query, history_turns=0)
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


def evaluate_group(name, rows, before_model, after_model, tokenizer, args):
    print()
    print(name)
    print("-" * len(name))
    results = []
    for row in rows:
        item = RoleProposition(
            subject=str(row["subject"]),
            relation=str(row["relation"]),
            object_description=str(row["object_description"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
        )
        query = semantic_role_prompt(item)
        before_answer = generate(before_model, tokenizer, query, args)
        after_answer = generate(after_model, tokenizer, query, args)
        before_score = score(
            before_model, tokenizer, before_answer, item.answer, item.subject
        )
        after_score = score(
            after_model, tokenizer, after_answer, item.answer, item.subject
        )
        gain = after_score - before_score
        label = "GAIN" if gain > 1e-6 else "SAME" if abs(gain) <= 1e-6 else "REGRESS"
        print(
            f"[{label}] role={item.relation:<10} subject={item.subject!r} "
            f"score={before_score:.3f}->{after_score:.3f} gain={gain:+.3f}"
        )
        results.append({
            "subject": item.subject,
            "relation": item.relation,
            "query": query,
            "teacher": item.answer,
            "before_answer": before_answer,
            "after_answer": after_answer,
            "before_score": before_score,
            "after_score": after_score,
            "gain": gain,
        })

    mean_gain = sum(x["gain"] for x in results) / len(results)
    improved = sum(1 for x in results if x["gain"] > 1e-6)
    same = sum(1 for x in results if abs(x["gain"]) <= 1e-6)
    regressed = len(results) - improved - same
    return results, mean_gain, improved, same, regressed


def main() -> None:
    args = parse_args()
    seen_rows = load_rows(Path(args.seen_holdout))
    unseen_rows = load_rows(Path(args.unseen_holdout))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    before_model, before_ckpt = LanguageModel.load_checkpoint(args.before, device=device)
    after_model, after_ckpt = LanguageModel.load_checkpoint(args.after, device=device)

    print("=" * 116)
    print(" LLM_TRY v10.12.11 Role-Balanced Semantic Evaluation")
    print("=" * 116)
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Before              :", args.before)
    print("After               :", args.after)
    print("Seen-role HOLDOUT   :", len(seen_rows))
    print("Unseen-role HOLDOUT :", len(unseen_rows))

    seen = evaluate_group(
        "A) Seen-role / unseen-concept",
        seen_rows,
        before_model,
        after_model,
        tokenizer,
        args,
    )
    unseen = evaluate_group(
        "B) Unseen-role",
        unseen_rows,
        before_model,
        after_model,
        tokenizer,
        args,
    )

    seen_results, seen_gain, seen_imp, seen_same, seen_reg = seen
    unseen_results, unseen_gain, unseen_imp, unseen_same, unseen_reg = unseen

    def relation_summary(rows):
        grouped = {}
        for row in rows:
            grouped.setdefault(row["relation"], []).append(row["gain"])
        return {
            relation: {
                "count": len(values),
                "mean_gain": sum(values) / len(values),
                "improved": sum(1 for x in values if x > 1e-6),
                "same": sum(1 for x in values if abs(x) <= 1e-6),
                "regressed": sum(1 for x in values if x < -1e-6),
            }
            for relation, values in sorted(grouped.items())
        }

    seen_by_relation = relation_summary(seen_results)
    unseen_by_relation = relation_summary(unseen_results)

    seen_ok = seen_gain >= args.min_seen_role_gain and seen_imp > seen_reg
    unseen_ok = unseen_gain >= args.min_unseen_role_gain and unseen_imp >= unseen_reg

    print()
    print("Semantic role summary")
    print("---------------------")
    print(f"Seen-role mean gain   : {seen_gain:+.3f}")
    print(f"Seen-role outcomes    : improved={seen_imp} same={seen_same} regressed={seen_reg}")
    print(f"Unseen-role mean gain : {unseen_gain:+.3f}")
    print(f"Unseen-role outcomes  : improved={unseen_imp} same={unseen_same} regressed={unseen_reg}")
    print("Seen-role status      :", "PASS" if seen_ok else "FAIL")
    print("Unseen-role status    :", "PASS" if unseen_ok else "FAIL")
    print()
    print("Per-relation gains")
    print("------------------")
    for relation, stats in seen_by_relation.items():
        print(
            f"seen/{relation:<10} count={stats['count']:2d} "
            f"mean={stats['mean_gain']:+.3f} "
            f"improved={stats['improved']} regressed={stats['regressed']}"
        )
    for relation, stats in unseen_by_relation.items():
        print(
            f"unseen/{relation:<8} count={stats['count']:2d} "
            f"mean={stats['mean_gain']:+.3f} "
            f"improved={stats['improved']} regressed={stats['regressed']}"
        )

    fp = set(checkpoint_trained_fingerprints(after_ckpt))
    protected = protected_internalized_records(
        Path(args.learning_log), fp, set(), set()
    )

    print()
    print("Post-role-SFT retention")
    print("-----------------------")
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
    metadata_ok = metadata.get("semantic_role_version") == "v10.12.11"

    final_ok = seen_ok and unseen_ok and retention_ok and metadata_ok

    print()
    print(
        "[semantic role summary: "
        f"seen_role={'PASS' if seen_ok else 'FAIL'}, "
        f"seen_gain={seen_gain:+.3f}, "
        f"unseen_role={'PASS' if unseen_ok else 'FAIL'}, "
        f"unseen_gain={unseen_gain:+.3f}, "
        f"retention={'PASS' if retention_ok else 'FAIL'}, "
        f"persona={'PASS' if persona_ok else 'FAIL'}, "
        f"metadata={'PASS' if metadata_ok else 'FAIL'}, "
        f"status={'PASS' if final_ok else 'FAIL'}]"
    )

    report = {
        "version": "v10.12.11",
        "before": args.before,
        "after": args.after,
        "seen_role_results": seen_results,
        "unseen_role_results": unseen_results,
        "seen_role_mean_gain": seen_gain,
        "seen_role_improved": seen_imp,
        "seen_role_same": seen_same,
        "seen_role_regressed": seen_reg,
        "seen_role_passed": seen_ok,
        "seen_relation_summary": seen_by_relation,
        "unseen_role_mean_gain": unseen_gain,
        "unseen_role_improved": unseen_imp,
        "unseen_role_same": unseen_same,
        "unseen_role_regressed": unseen_reg,
        "unseen_role_passed": unseen_ok,
        "unseen_relation_summary": unseen_by_relation,
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
    print("Report               :", report_path)

    if not final_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
