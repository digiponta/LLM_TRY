#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.8 Corpus-to-QA lightweight SFT.

Trains only on the TRAIN QA file produced by build_nagato_qa_split_v10128.py.
The HOLDOUT file is intentionally not accepted as a training argument.
"""

from __future__ import annotations

import argparse
import json
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
    p = argparse.ArgumentParser(
        description="Lightweight QA SFT for corpus-grounded Nagato training pairs."
    )
    p.add_argument("--train", default="data/nagato_qa_train_v10128.jsonl")
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    p.add_argument("--base-model", default="model/model-gpu-v1.6.2-online.pt")
    p.add_argument(
        "--output",
        default="model/model-gpu-v1.6.2-online-nagato-qa-candidate.pt",
    )
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--freeze-blocks", type=int, default=2)
    return p.parse_args()


def load_train_rows(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"TRAIN QA file not found: {path}")
    pairs: List[Tuple[str, str]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        split_reason = str(row.get("split_reason", ""))
        if "holdout" in split_reason:
            raise ValueError(
                f"{path}:{line_no}: HOLDOUT row found in TRAIN file."
            )
        if question and answer:
            pairs.append((question, answer))
    if len(pairs) < 2:
        raise ValueError("At least 2 TRAIN QA pairs are required.")
    return pairs


class QADataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        tokenizer: Tokenizer,
        context_length: int,
    ):
        self.rows = []
        for question, answer in pairs:
            prompt_ids = tokenizer.encode(
                f"{USER_PREFIX}{question}\n{AI_PREFIX}",
                add_bos=True,
            )
            answer_ids = tokenizer.encode(answer, add_eos=True)
            max_sequence = context_length + 1

            if len(prompt_ids) + len(answer_ids) > max_sequence:
                keep_prompt = max(1, max_sequence - len(answer_ids))
                prompt_ids = prompt_ids[-keep_prompt:]

            if len(prompt_ids) + len(answer_ids) > max_sequence:
                room = max_sequence - len(prompt_ids)
                answer_ids = answer_ids[:max(1, room)]
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

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def masked_loss(model, x, y, mask):
    logits = model(x)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        reduction="none",
    ).view_as(y)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


def main() -> None:
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_path = Path(args.train)
    tokenizer_path = Path(args.tokenizer)
    base_path = Path(args.base_model)
    output_path = Path(args.output)

    pairs = load_train_rows(train_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, checkpoint = LanguageModel.load_checkpoint(
        str(base_path), device=device
    )

    freeze_blocks = max(0, min(args.freeze_blocks, len(model.blocks)))
    for index in range(freeze_blocks):
        for parameter in model.blocks[index].parameters():
            parameter.requires_grad = False

    dataset = QADataset(pairs, tokenizer, model.context_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("=" * 104)
    print(" LLM_TRY v10.12.8 Corpus-to-QA Lightweight SFT")
    print("=" * 104)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Base model      :", base_path)
    print("Base loss       :", checkpoint.get("loss"))
    print("TRAIN file      :", train_path)
    print("TRAIN pairs     :", len(pairs))
    print("Frozen blocks   :", freeze_blocks)
    print("Learning rate   :", args.learning_rate)
    print("Epoch limit     :", args.epochs)
    print("Output          :", output_path)
    print()

    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        started = time.perf_counter()

        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model, x, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            total += float(loss.item())
            batches += 1

        train_loss = total / max(1, batches)
        elapsed = time.perf_counter() - started
        print(
            f"epoch={epoch:02d} train={train_loss:.6f} "
            f"time={elapsed:.2f}s"
        )
        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No valid candidate checkpoint produced.")

    model.load_state_dict(best_state)
    model.to(device)

    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    metadata.update({
        "corpus_to_qa_version": "v10.12.8",
        "corpus_to_qa_train": str(train_path),
        "corpus_to_qa_pairs": len(pairs),
        "corpus_to_qa_learning_rate": args.learning_rate,
        "corpus_to_qa_epochs": best_epoch,
        "corpus_to_qa_frozen_blocks": freeze_blocks,
        "corpus_to_qa_base_model": str(base_path),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(
        str(output_path),
        optimizer=optimizer,
        epoch=best_epoch,
        loss=best_loss,
        metadata=metadata,
    )

    print()
    print("Training completed.")
    print("Best epoch      :", best_epoch)
    print("Best train loss :", f"{best_loss:.6f}")
    print("Saved candidate :", output_path)


if __name__ == "__main__":
    main()
