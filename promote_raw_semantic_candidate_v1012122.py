#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.12.2 Raw+Semantic Candidate Promotion Gate."""

from __future__ import annotations

import argparse
import json
import shutil
import time
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
from semantic_role_generalization_v101210 import RoleProposition, semantic_role_prompt


DEFAULT_PRODUCTION = "model/model-gpu-v1.6.2-online.pt"
DEFAULT_CANDIDATE = "model/model-gpu-v1.6.2-online-nagato-semantic-candidate.pt"
DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_LOG = "data/chat_history.jsonl"
DEFAULT_SEEN = "data/nagato_corpus_semantic_holdout_seen_v101212.jsonl"
DEFAULT_UNSEEN = "data/nagato_corpus_semantic_holdout_unseen_v101212.jsonl"
DEFAULT_AUDIT = "data/raw_semantic_candidate_gate_v1012122.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate and optionally promote the v10.12.12.1 raw+semantic candidate."
    )
    p.add_argument("--production", default=DEFAULT_PRODUCTION)
    p.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--learning-log", default=DEFAULT_LOG)
    p.add_argument("--seen", default=DEFAULT_SEEN)
    p.add_argument("--unseen", default=DEFAULT_UNSEEN)
    p.add_argument("--audit", default=DEFAULT_AUDIT)
    p.add_argument("--promote", action="store_true")
    p.add_argument("--min-plain-gain", type=float, default=0.005)
    p.add_argument("--min-semantic-gain", type=float, default=0.010)
    p.add_argument("--min-seen-semantic-gain", type=float, default=0.010)
    p.add_argument("--min-unseen-semantic-gain", type=float, default=0.000)
    p.add_argument("--min-semantic", type=float, default=0.90)
    p.add_argument("--min-lexical", type=float, default=0.45)
    p.add_argument("--min-required", type=float, default=0.50)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    return p.parse_args()


def atomic_promote(candidate: Path, production: Path) -> None:
    backup = production.with_name(production.name + ".raw-semantic-promotion-backup")
    if backup.exists():
        backup.unlink()
    had_production = production.exists()
    if had_production:
        shutil.copy2(production, backup)
    try:
        candidate.replace(production)
    except Exception:
        if had_production and backup.exists():
            shutil.copy2(backup, production)
        elif not had_production and production.exists():
            production.unlink()
        raise
    finally:
        if backup.exists():
            backup.unlink()


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def generate_text(model, tokenizer, question: str, args) -> str:
    prompt, _ = build_prompt(history=[], user_text=question, history_turns=0)
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


def composite_score(model, tokenizer, generated: str, teacher: str, concept: str) -> float:
    result = composite_internalized_teacher_fidelity(
        model=model,
        tokenizer=tokenizer,
        generated_answer=generated,
        teacher_answer=teacher,
        concept=concept,
    )
    return (
        result.semantic_similarity
        + result.lexical_coverage
        + result.required_coverage
    ) / 3.0


def evaluate_holdout(rows, production_model, candidate_model, tokenizer, args):
    results = []
    for row in rows:
        item = RoleProposition(
            subject=str(row["subject"]),
            relation=str(row["relation"]),
            object_description=str(row["object_description"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
        )
        semantic_query = semantic_role_prompt(item)

        prod_plain = generate_text(production_model, tokenizer, item.question, args)
        cand_plain = generate_text(candidate_model, tokenizer, item.question, args)
        prod_sem = generate_text(production_model, tokenizer, semantic_query, args)
        cand_sem = generate_text(candidate_model, tokenizer, semantic_query, args)

        prod_plain_score = composite_score(
            production_model, tokenizer, prod_plain, item.answer, item.subject
        )
        cand_plain_score = composite_score(
            candidate_model, tokenizer, cand_plain, item.answer, item.subject
        )
        prod_sem_score = composite_score(
            production_model, tokenizer, prod_sem, item.answer, item.subject
        )
        cand_sem_score = composite_score(
            candidate_model, tokenizer, cand_sem, item.answer, item.subject
        )

        results.append({
            "subject": item.subject,
            "relation": item.relation,
            "plain_gain": cand_plain_score - prod_plain_score,
            "semantic_gain": cand_sem_score - prod_sem_score,
        })
    return results


def summarize(rows):
    return {
        "count": len(rows),
        "plain_gain": sum(x["plain_gain"] for x in rows) / len(rows),
        "semantic_gain": sum(x["semantic_gain"] for x in rows) / len(rows),
        "semantic_improved": sum(1 for x in rows if x["semantic_gain"] > 1e-6),
        "semantic_same": sum(1 for x in rows if abs(x["semantic_gain"]) <= 1e-6),
        "semantic_regressed": sum(1 for x in rows if x["semantic_gain"] < -1e-6),
    }


def main() -> None:
    args = parse_args()
    production_path = Path(args.production)
    candidate_path = Path(args.candidate)
    tokenizer_path = Path(args.tokenizer)
    learning_log = Path(args.learning_log)
    seen_path = Path(args.seen)
    unseen_path = Path(args.unseen)
    audit_path = Path(args.audit)

    for path, label in (
        (production_path, "Production checkpoint"),
        (candidate_path, "Raw+semantic candidate"),
        (tokenizer_path, "Tokenizer"),
        (seen_path, "Seen-role HOLDOUT"),
        (unseen_path, "Unseen-role HOLDOUT"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(str(tokenizer_path))
    production_model, production_ckpt = LanguageModel.load_checkpoint(
        str(production_path), device=device
    )
    candidate_model, candidate_ckpt = LanguageModel.load_checkpoint(
        str(candidate_path), device=device
    )

    production_fp = set(checkpoint_trained_fingerprints(production_ckpt))
    candidate_fp = set(checkpoint_trained_fingerprints(candidate_ckpt))
    binding_ok = production_fp.issubset(candidate_fp)

    metadata = candidate_ckpt.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    raw_metadata_ok = metadata.get("nagato_moderate_version") == "v10.12.4"
    semantic_metadata_ok = metadata.get("corpus_semantic_version") == "v10.12.12"
    pipeline_metadata_ok = (
        metadata.get("nagato_retraining_version") == "v10.12.11.1"
        or raw_metadata_ok
    )
    metadata_ok = raw_metadata_ok and semantic_metadata_ok and pipeline_metadata_ok

    protected = protected_internalized_records(
        learning_log,
        production_fp,
        set(),
        set(),
    )

    print("=" * 116)
    print(" LLM_TRY v10.12.12.2 Raw+Semantic Candidate Promotion Gate")
    print("=" * 116)
    print("Device                   :", device)
    if device.type == "cuda":
        print("GPU                      :", torch.cuda.get_device_name(0))
    print("Production               :", production_path)
    print("Candidate                :", candidate_path)
    print("Production binding       :", len(production_fp))
    print("Candidate binding        :", len(candidate_fp))
    print("Binding preserved        :", binding_ok)
    print("Raw metadata             :", raw_metadata_ok)
    print("Semantic metadata        :", semantic_metadata_ok)
    print("Pipeline metadata        :", pipeline_metadata_ok)
    print("Protected concepts       :", len(protected))
    print()

    probes = []
    for record in protected:
        generated = generate_text(
            candidate_model, tokenizer, record.question, args
        )
        malformed = malformed_or_unstable(generated)
        result = composite_internalized_teacher_fidelity(
            model=candidate_model,
            tokenizer=tokenizer,
            generated_answer=generated,
            teacher_answer=record.teacher_answer,
            concept=record.concept,
        )
        ok, reason = apply_composite_internalized_fidelity_policy(
            accepted=not malformed,
            result=result,
            semantic_threshold=args.min_semantic,
            lexical_threshold=args.min_lexical,
            required_threshold=args.min_required,
        )
        print(
            f"[{'PASS' if ok else 'FAIL'}] internalized={record.concept!r} "
            f"sem={result.semantic_similarity:.3f} "
            f"lex={result.lexical_coverage:.3f} "
            f"req={result.required_coverage:.3f}"
        )
        probes.append({
            "concept": record.concept,
            "passed": ok,
            "reason": reason,
            "semantic": result.semantic_similarity,
            "lexical": result.lexical_coverage,
            "required": result.required_coverage,
        })

    preservation_ok = len(probes) > 0 and all(x["passed"] for x in probes)

    identity = generate_text(
        candidate_model, tokenizer, "あなたは誰ですか", args
    )
    persona_ok = identity.strip().rstrip("。") == "長門有希"
    print(f"[{'PASS' if persona_ok else 'FAIL'}] persona candidate={identity!r}")

    seen_rows = load_rows(seen_path)
    unseen_rows = load_rows(unseen_path)
    seen_results = evaluate_holdout(
        seen_rows, production_model, candidate_model, tokenizer, args
    )
    unseen_results = evaluate_holdout(
        unseen_rows, production_model, candidate_model, tokenizer, args
    )
    all_results = seen_results + unseen_results

    seen_summary = summarize(seen_results)
    unseen_summary = summarize(unseen_results)
    overall_summary = summarize(all_results)

    plain_ok = overall_summary["plain_gain"] >= args.min_plain_gain
    semantic_ok = (
        overall_summary["semantic_gain"] >= args.min_semantic_gain
        and overall_summary["semantic_improved"]
        > overall_summary["semantic_regressed"]
    )
    seen_ok = (
        seen_summary["semantic_gain"] >= args.min_seen_semantic_gain
        and seen_summary["semantic_improved"] > seen_summary["semantic_regressed"]
    )
    unseen_ok = (
        unseen_summary["semantic_gain"] >= args.min_unseen_semantic_gain
        and unseen_summary["semantic_improved"] >= unseen_summary["semantic_regressed"]
    )

    print()
    print("Generalization gates")
    print("--------------------")
    print(
        f"Plain QA gain            : {overall_summary['plain_gain']:+.3f} "
        f"=> {'PASS' if plain_ok else 'FAIL'}"
    )
    print(
        f"Semantic gain            : {overall_summary['semantic_gain']:+.3f} "
        f"=> {'PASS' if semantic_ok else 'FAIL'}"
    )
    print(
        f"Seen semantic gain       : {seen_summary['semantic_gain']:+.3f} "
        f"=> {'PASS' if seen_ok else 'FAIL'}"
    )
    print(
        f"Unseen semantic gain     : {unseen_summary['semantic_gain']:+.3f} "
        f"=> {'PASS' if unseen_ok else 'FAIL'}"
    )
    print(
        "Semantic outcomes        : "
        f"improved={overall_summary['semantic_improved']} "
        f"same={overall_summary['semantic_same']} "
        f"regressed={overall_summary['semantic_regressed']}"
    )

    overall_ok = all((
        binding_ok,
        metadata_ok,
        preservation_ok,
        persona_ok,
        plain_ok,
        semantic_ok,
        seen_ok,
        unseen_ok,
    ))

    passed_internalized = sum(1 for x in probes if x["passed"])

    print()
    print(
        "[raw+semantic candidate gate: "
        f"binding={'PASS' if binding_ok else 'FAIL'}, "
        f"metadata={'PASS' if metadata_ok else 'FAIL'}, "
        f"internalized={passed_internalized}/{len(probes)}, "
        f"persona={'PASS' if persona_ok else 'FAIL'}, "
        f"plainQA={'PASS' if plain_ok else 'FAIL'}, "
        f"semantic={'PASS' if semantic_ok else 'FAIL'}, "
        f"seen={'PASS' if seen_ok else 'FAIL'}, "
        f"unseen={'PASS' if unseen_ok else 'FAIL'}, "
        f"status={'PASS' if overall_ok else 'FAIL'}]"
    )

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "version": "v10.12.12.2",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "production": str(production_path),
            "candidate": str(candidate_path),
            "binding_preserved": binding_ok,
            "raw_metadata": raw_metadata_ok,
            "semantic_metadata": semantic_metadata_ok,
            "pipeline_metadata": pipeline_metadata_ok,
            "internalized": probes,
            "persona_answer": identity,
            "persona_passed": persona_ok,
            "seen_summary": seen_summary,
            "unseen_summary": unseen_summary,
            "overall_summary": overall_summary,
            "plain_passed": plain_ok,
            "semantic_passed": semantic_ok,
            "seen_passed": seen_ok,
            "unseen_passed": unseen_ok,
            "status": "PASS" if overall_ok else "FAIL",
            "promote_requested": bool(args.promote),
        }, ensure_ascii=False) + "\n")

    if not overall_ok:
        print("[promotion blocked: raw+semantic candidate gate failed; production unchanged]")
        raise SystemExit(2)

    if not args.promote:
        print("[candidate accepted: evaluation only; production unchanged]")
        print("[run again with --promote to promote this candidate]")
        return

    atomic_promote(candidate_path, production_path)
    print(f"[raw+semantic candidate promoted: {production_path}]")
    print("[production checkpoint replaced only after all gates passed]")


if __name__ == "__main__":
    main()
