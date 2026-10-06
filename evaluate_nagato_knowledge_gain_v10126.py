#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.6 Nagato Knowledge Gain Evaluation.

Compares a pre-Nagato checkpoint and the current post-Nagato checkpoint on:
- held-out corpus NLL / perplexity,
- per-window improvement rate,
- optional QA probes,
- internalized-knowledge retention on the post model,
- persona retention.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

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


DEFAULT_DATA = "data/data-nagato.txt"
DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_AFTER = "model/model-gpu-v1.6.2-online.pt"
DEFAULT_BEFORE = "model/model-gpu-v1.6.2-online-pre-nagato.pt"
DEFAULT_LOG = "data/chat_history.jsonl"
DEFAULT_PROBES = "data/nagato_gain_probes_v10127.jsonl"
DEFAULT_REPORT = "results/nagato_knowledge_gain_v10127.json"


@dataclass(frozen=True)
class WindowScore:
    index: int
    before_nll: float
    after_nll: float

    @property
    def gain(self) -> float:
        return self.before_nll - self.after_nll


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare pre/post Nagato checkpoints and quantify corpus knowledge gain."
    )
    p.add_argument("--before", default=DEFAULT_BEFORE)
    p.add_argument("--after", default=DEFAULT_AFTER)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--learning-log", default=DEFAULT_LOG)
    p.add_argument("--probes", default=DEFAULT_PROBES)
    p.add_argument("--report", default=DEFAULT_REPORT)
    p.add_argument("--validation-ratio", type=float, default=0.10)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--min-semantic", type=float, default=0.90)
    p.add_argument("--min-lexical", type=float, default=0.45)
    p.add_argument("--min-required", type=float, default=0.50)
    p.add_argument(
        "--min-qa-mean-gain",
        type=float,
        default=0.01,
        help="Minimum mean composite QA score gain for v10.12.7 QA PASS.",
    )
    return p.parse_args()


def resolve_data_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    if value == DEFAULT_DATA:
        fallback = Path("data-nagato.txt")
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Nagato corpus not found: {path}")


def window_starts(length: int, block_size: int, stride: int) -> list[int]:
    if length < block_size + 1:
        raise ValueError(
            f"Not enough tokens ({length}) for block_size={block_size}."
        )
    last_start = length - (block_size + 1)
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


@torch.no_grad()
def sequence_nll(
    model: LanguageModel,
    token_ids: list[int],
    start: int,
    block_size: int,
    device: torch.device,
) -> float:
    seq = token_ids[start : start + block_size + 1]
    x = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([seq[1:]], dtype=torch.long, device=device)
    logits = model(x)
    return float(
        F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        ).item()
    )


def evaluate_windows(
    before_model: LanguageModel,
    after_model: LanguageModel,
    token_ids: list[int],
    starts: Iterable[int],
    block_size: int,
    device: torch.device,
) -> list[WindowScore]:
    scores: list[WindowScore] = []
    for index, start in enumerate(starts):
        before = sequence_nll(
            before_model, token_ids, start, block_size, device
        )
        after = sequence_nll(
            after_model, token_ids, start, block_size, device
        )
        scores.append(
            WindowScore(
                index=index,
                before_nll=before,
                after_nll=after,
            )
        )
    return scores


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


def load_probes(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if question and answer:
            rows.append({
                "question": question,
                "answer": answer,
                "concept": str(row.get("concept", "")).strip(),
                "probe_id": str(row.get("probe_id", "")).strip(),
                "source_text": str(row.get("source_text", "")).strip(),
                "source": str(row.get("source", "")).strip(),
            })
    return rows


def main() -> None:
    args = parse_args()
    before_path = Path(args.before)
    after_path = Path(args.after)
    tokenizer_path = Path(args.tokenizer)
    data_path = resolve_data_path(args.data)
    report_path = Path(args.report)

    if not before_path.exists():
        raise FileNotFoundError(
            f"Pre-Nagato checkpoint not found: {before_path}\n"
            "v10.12.5 promotion already replaced production. Supply an older "
            "checkpoint with --before PATH, or create a pre-Nagato snapshot "
            "before the next moderate-training run."
        )
    if not after_path.exists():
        raise FileNotFoundError(f"Post-Nagato checkpoint not found: {after_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(str(tokenizer_path))
    before_model, before_ckpt = LanguageModel.load_checkpoint(
        str(before_path), device=device
    )
    after_model, after_ckpt = LanguageModel.load_checkpoint(
        str(after_path), device=device
    )

    text = data_path.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    block_size = min(
        args.block_size,
        before_model.context_length,
        after_model.context_length,
    )

    split = max(
        block_size + 1,
        int(round(len(token_ids) * (1.0 - args.validation_ratio))),
    )
    split = min(split, len(token_ids) - (block_size + 1))
    heldout_ids = token_ids[split:]
    starts = window_starts(len(heldout_ids), block_size, args.stride)

    scores = evaluate_windows(
        before_model,
        after_model,
        heldout_ids,
        starts,
        block_size,
        device,
    )

    before_mean = sum(x.before_nll for x in scores) / len(scores)
    after_mean = sum(x.after_nll for x in scores) / len(scores)
    mean_gain = before_mean - after_mean
    relative_gain = (
        mean_gain / before_mean
        if before_mean > 0
        else 0.0
    )
    improved = sum(1 for x in scores if x.gain > 0)
    unchanged = sum(1 for x in scores if abs(x.gain) <= 1e-6)
    regressed = len(scores) - improved - unchanged

    before_ppl = math.exp(min(before_mean, 20.0))
    after_ppl = math.exp(min(after_mean, 20.0))

    print("=" * 112)
    print(" LLM_TRY v10.12.7 Nagato Knowledge Gain + QA Probe Evaluation")
    print("=" * 112)
    print("Device             :", device)
    if device.type == "cuda":
        print("GPU                :", torch.cuda.get_device_name(0))
    print("Before             :", before_path)
    print("After              :", after_path)
    print("Corpus             :", data_path)
    print("Corpus tokens      :", len(token_ids))
    print("Held-out tokens    :", len(heldout_ids))
    print("Held-out windows   :", len(scores))
    print("Block / stride     :", block_size, "/", args.stride)
    print()
    print("Held-out corpus")
    print("----------------")
    print(f"Before mean NLL    : {before_mean:.6f}")
    print(f"After mean NLL     : {after_mean:.6f}")
    print(f"NLL gain           : {mean_gain:+.6f}")
    print(f"Relative gain      : {relative_gain:+.2%}")
    print(f"Before perplexity  : {before_ppl:.3f}")
    print(f"After perplexity   : {after_ppl:.3f}")
    print(
        "Window outcomes    : "
        f"improved={improved} "
        f"unchanged={unchanged} "
        f"regressed={regressed}"
    )

    probe_rows = load_probes(Path(args.probes))
    probe_results = []
    if probe_rows:
        print()
        print("Optional QA probes")
        print("------------------")
    for row in probe_rows:
        question = row["question"]
        teacher = row["answer"]
        concept = row["concept"]
        before_answer = generate_text(
            before_model, tokenizer, question, args
        )
        after_answer = generate_text(
            after_model, tokenizer, question, args
        )
        before_fid = composite_internalized_teacher_fidelity(
            model=before_model,
            tokenizer=tokenizer,
            generated_answer=before_answer,
            teacher_answer=teacher,
            concept=concept,
        )
        after_fid = composite_internalized_teacher_fidelity(
            model=after_model,
            tokenizer=tokenizer,
            generated_answer=after_answer,
            teacher_answer=teacher,
            concept=concept,
        )
        before_score = (
            before_fid.semantic_similarity
            + before_fid.lexical_coverage
            + before_fid.required_coverage
        ) / 3.0
        after_score = (
            after_fid.semantic_similarity
            + after_fid.lexical_coverage
            + after_fid.required_coverage
        ) / 3.0
        gain = after_score - before_score
        print(
            f"[{'GAIN' if gain > 0 else 'SAME' if abs(gain) <= 1e-6 else 'REGRESS'}] "
            f"{question!r} score={before_score:.3f}->{after_score:.3f} "
            f"gain={gain:+.3f}"
        )
        probe_results.append({
            **row,
            "before_answer": before_answer,
            "after_answer": after_answer,
            "before_score": before_score,
            "after_score": after_score,
            "gain": gain,
        })

    production_fp = set(checkpoint_trained_fingerprints(after_ckpt))
    protected = protected_internalized_records(
        Path(args.learning_log),
        production_fp,
        set(),
        set(),
    )

    print()
    print("Post-Nagato retention")
    print("---------------------")
    retention_results = []
    for record in protected:
        answer = generate_text(
            after_model, tokenizer, record.question, args
        )
        malformed = malformed_or_unstable(answer)
        result = composite_internalized_teacher_fidelity(
            model=after_model,
            tokenizer=tokenizer,
            generated_answer=answer,
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
            f"[{'PASS' if ok else 'FAIL'}] {record.concept!r} "
            f"sem={result.semantic_similarity:.3f} "
            f"lex={result.lexical_coverage:.3f} "
            f"req={result.required_coverage:.3f}"
        )
        retention_results.append({
            "concept": record.concept,
            "passed": ok,
            "answer": answer,
            "semantic": result.semantic_similarity,
            "lexical": result.lexical_coverage,
            "required": result.required_coverage,
            "reason": reason,
        })

    identity = generate_text(
        after_model,
        tokenizer,
        "あなたは誰ですか",
        args,
    )
    persona_ok = identity.strip().rstrip("。") == "長門有希"
    retention_ok = all(x["passed"] for x in retention_results) and persona_ok

    corpus_gain_ok = mean_gain > 0 and improved > regressed
    qa_gain_mean = (
        sum(x["gain"] for x in probe_results) / len(probe_results)
        if probe_results
        else None
    )
    qa_improved = sum(1 for x in probe_results if x["gain"] > 1e-6)
    qa_same = sum(1 for x in probe_results if abs(x["gain"]) <= 1e-6)
    qa_regressed = len(probe_results) - qa_improved - qa_same
    qa_gain_ok = (
        bool(probe_results)
        and qa_gain_mean is not None
        and qa_gain_mean >= args.min_qa_mean_gain
        and qa_improved > qa_regressed
    )

    if probe_results:
        print()
        print("QA probe summary")
        print("----------------")
        print(f"Probe count         : {len(probe_results)}")
        print(f"Improved            : {qa_improved}")
        print(f"Same                : {qa_same}")
        print(f"Regressed           : {qa_regressed}")
        print(f"Mean QA gain        : {qa_gain_mean:+.3f}")
        print(
            "QA gain status      :",
            "PASS" if qa_gain_ok else "FAIL",
        )

    print()
    print(
        "[knowledge gain summary: "
        f"corpus_gain={'PASS' if corpus_gain_ok else 'FAIL'}, "
        f"nll_gain={mean_gain:+.6f}, "
        f"window_improved={improved}/{len(scores)}, "
        f"qa_gain={'n/a' if qa_gain_mean is None else f'{qa_gain_mean:+.3f}'}, "
        f"qa_status={'n/a' if not probe_results else ('PASS' if qa_gain_ok else 'FAIL')}, "
        f"retention={'PASS' if retention_ok else 'FAIL'}, "
        f"persona={'PASS' if persona_ok else 'FAIL'}]"
    )

    report = {
        "version": "v10.12.7",
        "before": str(before_path),
        "after": str(after_path),
        "data": str(data_path),
        "corpus_tokens": len(token_ids),
        "heldout_tokens": len(heldout_ids),
        "heldout_windows": len(scores),
        "before_mean_nll": before_mean,
        "after_mean_nll": after_mean,
        "nll_gain": mean_gain,
        "relative_gain": relative_gain,
        "before_perplexity": before_ppl,
        "after_perplexity": after_ppl,
        "windows_improved": improved,
        "windows_unchanged": unchanged,
        "windows_regressed": regressed,
        "corpus_gain_passed": corpus_gain_ok,
        "qa_probes": probe_results,
        "qa_gain_mean": qa_gain_mean,
        "qa_improved": qa_improved,
        "qa_same": qa_same,
        "qa_regressed": qa_regressed,
        "qa_gain_passed": qa_gain_ok,
        "qa_gain_threshold": args.min_qa_mean_gain,
        "retention": retention_results,
        "retention_passed": retention_ok,
        "persona_answer": identity,
        "persona_passed": persona_ok,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Report             :", report_path)


if __name__ == "__main__":
    main()
