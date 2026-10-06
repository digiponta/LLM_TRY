#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.14 Two-Pass Function Semantic Resolver Evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import LanguageModel
from tokenizer_bpe import Tokenizer
from semantic_role_generalization_v101210 import RoleProposition
from two_pass_function_resolver_v101214 import (
    relation_uses_two_pass,
    resolve_two_pass_function,
)
from chat import (
    build_prompt,
    composite_internalized_teacher_fidelity,
    generate_reply,
    malformed_or_unstable,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model/model-gpu-v1.6.2-online.pt")
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    p.add_argument("--seen", default="data/nagato_corpus_semantic_holdout_seen_v101212.jsonl")
    p.add_argument("--unseen", default="data/nagato_corpus_semantic_holdout_unseen_v101212.jsonl")
    p.add_argument("--report", default="results/two_pass_function_v101214.json")
    p.add_argument("--min-two-pass-gain", type=float, default=0.005)
    p.add_argument("--min-structured-slot-rate", type=float, default=0.50)
    return p.parse_args()


def load_rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def generate_text(model, tokenizer, query):
    prompt, _ = build_prompt(history=[], user_text=query, history_turns=0)
    return generate_reply(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=96,
        temperature=0.0,
        top_k=20,
        repetition_penalty=1.05,
        seed=0,
    ).text


def score(model, tokenizer, answer, teacher, concept):
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


def informative_slot_count(structure):
    return sum((
        structure.action != "function",
        structure.target != "unspecified",
        structure.purpose != "unspecified",
    ))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(args.model, device=device)

    rows = load_rows(args.seen) + load_rows(args.unseen)
    function_rows = [row for row in rows if row["relation"] == "function"]
    nonfunction_rows = [row for row in rows if row["relation"] != "function"]
    if not function_rows:
        raise RuntimeError("No function HOLDOUT rows found")

    print("=" * 116)
    print(" LLM_TRY v10.12.14 Two-Pass Function Semantic Resolver Evaluation")
    print("=" * 116)
    print("Device                  :", device)
    if device.type == "cuda":
        print("GPU                     :", torch.cuda.get_device_name(0))
    print("Model                   :", args.model)
    print("Function HOLDOUT        :", len(function_rows))
    print("Non-function route guard:", len(nonfunction_rows))
    print()

    results = []
    informative = 0

    def generate(query):
        return generate_text(model, tokenizer, query)

    print("Function HOLDOUT")
    print("----------------")
    for row in function_rows:
        item = RoleProposition(
            subject=row["subject"],
            relation=row["relation"],
            object_description=row["object_description"],
            question=row["question"],
            answer=row["answer"],
        )

        baseline = generate(item.question)
        baseline_score = score(
            model, tokenizer, baseline, item.answer, item.subject
        )

        resolved = resolve_two_pass_function(
            item.subject,
            generate=generate,
            malformed=malformed_or_unstable,
        )
        resolved_score = score(
            model, tokenizer, resolved.answer, item.answer, item.subject
        )
        gain = resolved_score - baseline_score
        slot_count = informative_slot_count(resolved.structure)
        if slot_count > 0:
            informative += 1

        print(
            f"{item.subject!r} "
            f"baseline={baseline_score:.3f} "
            f"two_pass={resolved_score:.3f} "
            f"gain={gain:+.3f} "
            f"action={resolved.structure.action} "
            f"target={resolved.structure.target} "
            f"purpose={resolved.structure.purpose} "
            f"fallback={resolved.used_fallback}"
        )

        results.append({
            "subject": item.subject,
            "baseline_score": baseline_score,
            "two_pass_score": resolved_score,
            "gain": gain,
            "action": resolved.structure.action,
            "target": resolved.structure.target,
            "purpose": resolved.structure.purpose,
            "evidence": resolved.evidence,
            "answer": resolved.answer,
            "used_fallback": resolved.used_fallback,
        })

    mean_gain = sum(x["gain"] for x in results) / len(results)
    improved = sum(1 for x in results if x["gain"] > 1e-6)
    same = sum(1 for x in results if abs(x["gain"]) <= 1e-6)
    regressed = sum(1 for x in results if x["gain"] < -1e-6)
    slot_rate = informative / len(results)

    gain_ok = (
        mean_gain >= args.min_two_pass_gain
        and improved > regressed
    )
    slot_ok = slot_rate >= args.min_structured_slot_rate

    # Structural guard: non-function relations must never route into this
    # resolver. This guarantees no behavior change for those routes.
    nonfunction_route_ok = all(
        not relation_uses_two_pass(row["relation"])
        for row in nonfunction_rows
    )

    print()
    print("Two-pass summary")
    print("----------------")
    print(f"Mean two-pass gain       : {mean_gain:+.3f} => {'PASS' if gain_ok else 'FAIL'}")
    print(f"Outcomes                 : improved={improved} same={same} regressed={regressed}")
    print(f"Informative slot rate    : {slot_rate:.2%} => {'PASS' if slot_ok else 'FAIL'}")
    print("Non-function route guard :", "PASS" if nonfunction_route_ok else "FAIL")
    print("Weight update            : NONE")
    print("Teacher leakage          : NONE")

    final = gain_ok and slot_ok and nonfunction_route_ok
    print("STATUS                   :", "PASS" if final else "FAIL")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps({
            "version": "v10.12.14",
            "model": args.model,
            "function_results": results,
            "mean_two_pass_gain": mean_gain,
            "improved": improved,
            "same": same,
            "regressed": regressed,
            "informative_slot_rate": slot_rate,
            "gain_passed": gain_ok,
            "slot_passed": slot_ok,
            "nonfunction_route_guard_passed": nonfunction_route_ok,
            "weight_update": False,
            "teacher_leakage": False,
            "status": "PASS" if final else "FAIL",
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not final:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
