#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.5 Nagato Candidate Preservation / Promotion Gate."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from model import LanguageModel
from tokenizer_bpe import Tokenizer
from internalized_knowledge_v10100 import load_internalized_records
from batch_repair_preservation_v10121 import protected_internalized_records
from chat import (
    apply_composite_internalized_fidelity_policy,
    build_prompt,
    checkpoint_trained_fingerprints,
    composite_internalized_teacher_fidelity,
    generate_reply,
    malformed_or_unstable,
)


DEFAULT_PRODUCTION = "model/model-gpu-v1.6.2-online.pt"
DEFAULT_CANDIDATE = "model/model-gpu-v1.6.2-online-nagato-candidate.pt"
DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_LOG = "data/chat_history.jsonl"
DEFAULT_AUDIT = "data/nagato_candidate_gate_v10125.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate and optionally promote the v10.12.4 Nagato candidate."
    )
    p.add_argument("--production", default=DEFAULT_PRODUCTION)
    p.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--learning-log", default=DEFAULT_LOG)
    p.add_argument("--audit", default=DEFAULT_AUDIT)
    p.add_argument("--promote", action="store_true")
    p.add_argument("--min-semantic", type=float, default=0.90)
    p.add_argument("--min-lexical", type=float, default=0.45)
    p.add_argument("--min-required", type=float, default=0.50)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    return p.parse_args()


def atomic_promote(candidate: Path, production: Path) -> None:
    backup = production.with_name(production.name + ".nagato-promotion-backup")
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


def main() -> None:
    args = parse_args()
    production_path = Path(args.production)
    candidate_path = Path(args.candidate)
    tokenizer_path = Path(args.tokenizer)
    learning_log = Path(args.learning_log)
    audit_path = Path(args.audit)

    if not production_path.exists():
        raise FileNotFoundError(f"Production checkpoint not found: {production_path}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"Nagato candidate not found: {candidate_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

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
    adaptation_ok = metadata.get("nagato_moderate_version") == "v10.12.4"

    protected = protected_internalized_records(
        learning_log,
        production_fp,
        set(),
        set(),
    )

    print("=" * 108)
    print(" LLM_TRY v10.12.5 Nagato Candidate Preservation / Promotion Gate")
    print("=" * 108)
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Production          :", production_path)
    print("Candidate           :", candidate_path)
    print("Production binding  :", len(production_fp))
    print("Candidate binding   :", len(candidate_fp))
    print("Binding preserved   :", binding_ok)
    print("Nagato metadata     :", adaptation_ok)
    print("Protected concepts  :", len(protected))
    print()

    probes = []
    for record in protected:
        generated = generate_text(
            candidate_model,
            tokenizer,
            record.question,
            args,
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
        if malformed and not reason:
            reason = "malformed/repetitive output"

        print(
            f"[{'PASS' if ok else 'FAIL'}] internalized={record.concept!r} "
            f"sem={result.semantic_similarity:.3f} "
            f"lex={result.lexical_coverage:.3f} "
            f"req={result.required_coverage:.3f} "
            f"contra={result.contradiction}"
        )
        if not ok:
            print("  candidate:", generated)
            print("  reason   :", reason)

        probes.append({
            "concept": record.concept,
            "question": record.question,
            "fingerprint": record.fingerprint,
            "generated": generated,
            "semantic": result.semantic_similarity,
            "lexical": result.lexical_coverage,
            "required": result.required_coverage,
            "contradiction": result.contradiction,
            "passed": ok,
            "reason": reason,
        })

    identity = generate_text(
        candidate_model,
        tokenizer,
        "あなたは誰ですか",
        args,
    )
    identity_ok = identity.strip().rstrip("。") == "長門有希"
    print(
        f"[{'PASS' if identity_ok else 'FAIL'}] persona "
        f"candidate={identity!r}"
    )

    preservation_ok = all(row["passed"] for row in probes)
    overall_ok = (
        binding_ok
        and adaptation_ok
        and preservation_ok
        and identity_ok
        and len(probes) > 0
    )

    print()
    print(
        "[nagato candidate gate: "
        f"binding={'PASS' if binding_ok else 'FAIL'}, "
        f"metadata={'PASS' if adaptation_ok else 'FAIL'}, "
        f"internalized={sum(1 for x in probes if x['passed'])}/{len(probes)}, "
        f"persona={'PASS' if identity_ok else 'FAIL'}, "
        f"status={'PASS' if overall_ok else 'FAIL'}]"
    )

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "version": "v10.12.5",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "production": str(production_path),
            "candidate": str(candidate_path),
            "production_binding": len(production_fp),
            "candidate_binding": len(candidate_fp),
            "binding_preserved": binding_ok,
            "adaptation_metadata": adaptation_ok,
            "persona_answer": identity,
            "persona_passed": identity_ok,
            "internalized_probes": probes,
            "status": "PASS" if overall_ok else "FAIL",
            "promote_requested": bool(args.promote),
        }, ensure_ascii=False) + "\n")

    if not overall_ok:
        print("[promotion blocked: candidate gate failed; production unchanged]")
        raise SystemExit(2)

    if not args.promote:
        print("[candidate accepted: evaluation only; production unchanged]")
        print("[run again with --promote to promote this candidate]")
        return

    atomic_promote(candidate_path, production_path)
    print(f"[nagato candidate promoted: {production_path}]")
    print("[production checkpoint replaced only after all gates passed]")


if __name__ == "__main__":
    main()
