# train_nagato_chat.py
#
# LLM_TRY Nagato Chat SFT
#
# Supervised fine-tuning on question/answer pairs stored as JSONL:
#   {"user":"...", "assistant":"..."}
#
# Default flow:
#   model-gpu-v0.8-chat-clean.pt
#        -> optional raw Nagato continued pretraining
#        -> model-llm-try-nagato.pt
#        -> this conversational SFT
#        -> model-llm-try-nagato-chat.pt
#
# Loss is computed only on assistant-answer tokens.

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import LanguageModel
from tokenizer_bpe import Tokenizer


USER_PREFIX = "人: "
AI_PREFIX = "AI: "
SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nagato-style conversational SFT for LLM_TRY.")
    p.add_argument("--data", default="data/nagato_chat.jsonl")
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    p.add_argument("--base-model", default="model/model-llm-try-nagato.pt")
    p.add_argument("--output", default="model/model-llm-try-nagato-chat.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--validation-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each SFT pair this many times in training.",
    )
    p.add_argument(
        "--trainable-blocks",
        type=int,
        default=2,
        help="Number of final Transformer blocks to fine-tune. Embedding and LM head stay frozen by default.",
    )
    p.add_argument(
        "--train-lm-head",
        action="store_true",
        help="Also fine-tune the LM head. Off by default to reduce catastrophic forgetting.",
    )
    return p.parse_args()


def load_pairs(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"SFT data not found: {path}")

    pairs: List[Tuple[str, str]] = []
    seen = set()

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e

        user = str(row.get("user", "")).strip()
        answer = str(row.get("assistant", "")).strip()

        if not user or not answer:
            continue

        key = (user, answer)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)

    if len(pairs) < 2:
        raise ValueError("At least 2 unique SFT pairs are required.")

    return pairs


class ConversationDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        tokenizer: Tokenizer,
        context_length: int,
    ):
        self.rows = []

        for user_text, answer_text in pairs:
            prompt_ids = tokenizer.encode(
                f"{USER_PREFIX}{user_text}\n{AI_PREFIX}",
                add_bos=True,
            )
            answer_ids = tokenizer.encode(answer_text, add_eos=True)

            max_sequence = context_length + 1

            if len(prompt_ids) + len(answer_ids) > max_sequence:
                keep_prompt = max(1, max_sequence - len(answer_ids))
                prompt_ids = prompt_ids[-keep_prompt:]

            if len(prompt_ids) + len(answer_ids) > max_sequence:
                room = max_sequence - len(prompt_ids)
                answer_ids = answer_ids[:max(1, room)]
                if answer_ids:
                    answer_ids[-1] = tokenizer.eos_id

            sequence = prompt_ids + answer_ids
            answer_start = len(prompt_ids)

            x = sequence[:-1]
            y = sequence[1:]

            mask = [
                1.0 if (i + 1) >= answer_start else 0.0
                for i in range(len(y))
            ]

            pad_count = context_length - len(x)
            x += [tokenizer.pad_id] * pad_count
            y += [tokenizer.pad_id] * pad_count
            mask += [0.0] * pad_count

            self.rows.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(mask, dtype=torch.float32),
            ))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def masked_loss(model, x, y, mask):
    logits = model(x)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        reduction="none",
    ).view_as(y)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    count = 0

    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        total += float(masked_loss(model, x, y, mask).item())
        count += 1

    return total / max(1, count)


def main() -> None:
    args = parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path(args.data)
    tokenizer_path = Path(args.tokenizer)
    base_path = Path(args.base_model)
    output_path = Path(args.output)

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    if not base_path.exists():
        fallback = Path("model/model-gpu-v0.8-chat-clean.pt")
        if args.base_model == "model/model-llm-try-nagato.pt" and fallback.exists():
            print(f"[WARN] {base_path} not found; falling back to {fallback}")
            base_path = fallback
        else:
            raise FileNotFoundError(f"Base model not found: {base_path}")

    pairs = load_pairs(data_path)
    random.shuffle(pairs)

    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, checkpoint = LanguageModel.load_checkpoint(str(base_path), device=device)

    if tokenizer.vocab_size != model.vocab_size:
        raise ValueError(
            f"Tokenizer/model vocabulary mismatch: "
            f"{tokenizer.vocab_size} != {model.vocab_size}"
        )

    # Small-data SFT: freeze most of the model to preserve the base language
    # ability.  Only the final N Transformer blocks and final LayerNorm are
    # adapted.  The LM head remains frozen unless explicitly requested.
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_blocks = max(1, min(args.trainable_blocks, len(model.blocks)))
    first_trainable = len(model.blocks) - trainable_blocks

    for block in model.blocks[first_trainable:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    for parameter in model.final_norm.parameters():
        parameter.requires_grad = True

    if args.train_lm_head:
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True

    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    val_count = max(1, int(round(len(pairs) * args.validation_ratio)))
    val_count = min(val_count, len(pairs) - 1)

    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    repeated_train: List[Tuple[str, str]] = []
    for pair in train_pairs:
        repeated_train.extend([pair] * max(1, args.repeat))

    random.shuffle(repeated_train)

    train_ds = ConversationDataset(
        repeated_train,
        tokenizer,
        model.context_length,
    )
    val_ds = ConversationDataset(
        val_pairs,
        tokenizer,
        model.context_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("=" * 76)
    print(" LLM_TRY Nagato Chat SFT")
    print("=" * 76)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Data            :", data_path)
    print("Unique pairs    :", len(pairs))
    print("Train pairs     :", len(train_pairs))
    print("Validation pairs:", len(val_pairs))
    print("Repeat          :", args.repeat)
    print("Train rows      :", len(train_ds))
    print("Tokenizer       :", tokenizer_path)
    print("Vocabulary      :", tokenizer.vocab_size)
    print("Base model      :", base_path)
    print("Base loss       :", checkpoint.get("loss"))
    print("Parameters      :", f"{model.parameter_count:,}")
    print("Trainable params:", f"{trainable_parameters:,}")
    print("Trainable blocks:", f"{first_trainable + 1}-{len(model.blocks)}")
    print("LM head train   :", args.train_lm_head)
    print("Context length  :", model.context_length)
    print("Learning rate   :", args.learning_rate)
    print("Epoch limit     :", args.epochs)
    print()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        started = time.perf_counter()

        for x, y, mask in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model, x, y, mask)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )

            optimizer.step()

            total += float(loss.item())
            batches += 1

        train_loss = total / max(1, batches)
        val_loss = evaluate(model, val_loader, device)
        elapsed = time.perf_counter() - started

        print(
            f"epoch={epoch:02d} "
            f"train={train_loss:.6f} "
            f"val={val_loss:.6f} "
            f"ppl={math.exp(min(val_loss, 20.0)):.2f} "
            f"time={elapsed:.2f}s"
        )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break

    if best_state is None:
        raise RuntimeError("No valid checkpoint produced.")

    model.load_state_dict(best_state)
    model.to(device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(
        str(output_path),
        optimizer=optimizer,
        epoch=best_epoch,
        loss=best_val,
    )

    print()
    print("Training completed.")
    print("Best epoch      :", best_epoch)
    print("Best val loss   :", f"{best_val:.6f}")
    print("Saved checkpoint:", output_path)
    print()
    print("Test:")
    print(
        "  python chat.py "
        f"--model {output_path} "
        f"--tokenizer {tokenizer_path}"
    )


if __name__ == "__main__":
    main()
