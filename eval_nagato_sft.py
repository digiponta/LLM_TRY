# eval_nagato_sft.py
#
# Diagnostic for LLM_TRY Nagato Chat SFT.
#
# Compares teacher-forced target likelihood for the clean base model and
# the Nagato Chat SFT model on both train and independent paraphrase sets.
#
# Metrics:
#   - assistant-token NLL
#   - perplexity
#   - target next-token top-1 accuracy
#   - first assistant token rank
#
# This tells us whether SFT is actually moving probability mass toward the
# desired Nagato answers even when free generation still looks unchanged.

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer_bpe import Tokenizer


USER_PREFIX = "人: "
AI_PREFIX = "AI: "


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose Nagato Chat SFT target learning.")
    p.add_argument("--train-data", default="data/nagato_chat.jsonl")
    p.add_argument("--val-data", default="data/nagato_chat_val.jsonl")
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    p.add_argument("--base-model", default="model/model-gpu-v0.8-chat-clean.pt")
    p.add_argument("--sft-model", default="model/model-llm-try-nagato-chat-cleanbase.pt")
    return p.parse_args()


def load_pairs(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
        user = str(row.get("user", "")).strip()
        answer = str(row.get("assistant", "")).strip()
        if user and answer:
            rows.append((user, answer))
    return rows


@torch.no_grad()
def evaluate_pair(model, tokenizer, user: str, answer: str, device):
    prompt_ids = tokenizer.encode(
        f"{USER_PREFIX}{user}\n{AI_PREFIX}",
        add_bos=True,
    )
    answer_ids = tokenizer.encode(answer, add_eos=True)

    sequence = prompt_ids + answer_ids
    if len(sequence) > model.context_length + 1:
        keep_prompt = max(1, model.context_length + 1 - len(answer_ids))
        prompt_ids = prompt_ids[-keep_prompt:]
        sequence = prompt_ids + answer_ids

    x = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([sequence[1:]], dtype=torch.long, device=device)

    logits = model(x)[0]
    answer_start = len(prompt_ids) - 1
    answer_logits = logits[answer_start:]
    answer_targets = y[0, answer_start:]

    losses = F.cross_entropy(
        answer_logits,
        answer_targets,
        reduction="none",
    )

    preds = torch.argmax(answer_logits, dim=-1)
    top1 = float((preds == answer_targets).float().mean().item())

    first_logits = answer_logits[0]
    first_target = int(answer_targets[0].item())
    target_score = first_logits[first_target]
    first_rank = int((first_logits > target_score).sum().item()) + 1

    return {
        "nll": float(losses.mean().item()),
        "top1": top1,
        "first_rank": first_rank,
    }


@torch.no_grad()
def evaluate_set(model, tokenizer, pairs, device):
    rows = [
        evaluate_pair(model, tokenizer, q, a, device)
        for q, a in pairs
    ]

    mean_nll = sum(r["nll"] for r in rows) / len(rows)
    mean_top1 = sum(r["top1"] for r in rows) / len(rows)
    mean_rank = sum(r["first_rank"] for r in rows) / len(rows)
    first_top1 = sum(1 for r in rows if r["first_rank"] == 1) / len(rows)

    return {
        "nll": mean_nll,
        "ppl": math.exp(min(mean_nll, 20.0)),
        "token_top1": mean_top1,
        "mean_first_rank": mean_rank,
        "first_top1": first_top1,
        "rows": rows,
    }


def print_summary(label, result):
    print(f"{label}")
    print("-" * 72)
    print(f"NLL                 : {result['nll']:.6f}")
    print(f"Perplexity          : {result['ppl']:.2f}")
    print(f"Target token top-1  : {result['token_top1'] * 100:.2f}%")
    print(f"First token top-1   : {result['first_top1'] * 100:.2f}%")
    print(f"Mean first-token rank: {result['mean_first_rank']:.2f}")
    print()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.load(args.tokenizer)
    train_pairs = load_pairs(Path(args.train_data))
    val_pairs = load_pairs(Path(args.val_data))

    base, _ = LanguageModel.load_checkpoint(args.base_model, device=device)
    sft, _ = LanguageModel.load_checkpoint(args.sft_model, device=device)

    print("=" * 72)
    print(" LLM_TRY Nagato Chat SFT Diagnostic")
    print("=" * 72)
    print("Device     :", device)
    if device.type == "cuda":
        print("GPU        :", torch.cuda.get_device_name(0))
    print("Train pairs:", len(train_pairs))
    print("Val pairs  :", len(val_pairs))
    print("Base model :", args.base_model)
    print("SFT model  :", args.sft_model)
    print()

    base_train = evaluate_set(base, tokenizer, train_pairs, device)
    sft_train = evaluate_set(sft, tokenizer, train_pairs, device)
    base_val = evaluate_set(base, tokenizer, val_pairs, device)
    sft_val = evaluate_set(sft, tokenizer, val_pairs, device)

    print_summary("BASE / TRAIN", base_train)
    print_summary("SFT / TRAIN", sft_train)
    print_summary("BASE / PARAPHRASE VAL", base_val)
    print_summary("SFT / PARAPHRASE VAL", sft_val)

    print("Delta")
    print("-" * 72)
    print(f"Train NLL delta      : {sft_train['nll'] - base_train['nll']:+.6f}")
    print(f"Val NLL delta        : {sft_val['nll'] - base_val['nll']:+.6f}")
    print(
        "Train token top-1   : "
        f"{base_train['token_top1'] * 100:.2f}% -> "
        f"{sft_train['token_top1'] * 100:.2f}%"
    )
    print(
        "Val token top-1     : "
        f"{base_val['token_top1'] * 100:.2f}% -> "
        f"{sft_val['token_top1'] * 100:.2f}%"
    )
    print(
        "Train first top-1   : "
        f"{base_train['first_top1'] * 100:.2f}% -> "
        f"{sft_train['first_top1'] * 100:.2f}%"
    )
    print(
        "Val first top-1     : "
        f"{base_val['first_top1'] * 100:.2f}% -> "
        f"{sft_val['first_top1'] * 100:.2f}%"
    )

    print()
    print("Selected identity/style probes")
    print("-" * 72)
    wanted = {
        "あなたは誰ですか",
        "自己紹介してください",
        "本は好きですか",
        "人間についてどう思いますか",
        "人工知能とは何ですか",
        "LLMとは何ですか",
        "量子力学とは何ですか",
    }
    for i, (q, a) in enumerate(train_pairs):
        if q not in wanted:
            continue
        b = base_train["rows"][i]
        s = sft_train["rows"][i]
        print(q)
        print("  target:", a)
        print(
            f"  base: nll={b['nll']:.4f} "
            f"top1={b['top1']*100:.1f}% first_rank={b['first_rank']}"
        )
        print(
            f"  sft : nll={s['nll']:.4f} "
            f"top1={s['top1']*100:.1f}% first_rank={s['first_rank']}"
        )


if __name__ == "__main__":
    main()
