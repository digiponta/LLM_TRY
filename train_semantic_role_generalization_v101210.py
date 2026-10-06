#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.10 Semantic Role Generalization trainer."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import LanguageModel
from tokenizer_bpe import Tokenizer
from semantic_role_generalization_v101210 import RoleProposition, training_queries


USER_PREFIX = "人: "
AI_PREFIX = "AI: "
SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train multi-role semantic QA prompts on TRAIN roles only."
    )
    p.add_argument("--train", default="data/nagato_role_train_v101210.jsonl")
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    p.add_argument("--base-model", default="model/model-gpu-v1.6.2-online.pt")
    p.add_argument(
        "--output",
        default="model/model-gpu-v1.6.2-online-semantic-role-candidate.pt",
    )
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--freeze-blocks", type=int, default=2)
    return p.parse_args()


def load_augmented_pairs(path: Path) -> tuple[List[Tuple[str, str]], set[str], int]:
    if not path.exists():
        raise FileNotFoundError(path)

    pairs: List[Tuple[str, str]] = []
    relations: set[str] = set()
    subjects: set[str] = set()

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        reason = str(row.get("split_reason", ""))
        if "holdout" in reason:
            raise ValueError(f"{path}:{line_no}: HOLDOUT leakage detected")

        subject = str(row.get("subject", "")).strip()
        relation = str(row.get("relation", "")).strip()
        object_description = str(row.get("object_description", "")).strip()
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not all((subject, relation, object_description, question, answer)):
            continue

        item = RoleProposition(
            subject=subject,
            relation=relation,
            object_description=object_description,
            question=question,
            answer=answer,
        )
        subjects.add(subject)
        relations.add(relation)
        for query in training_queries(item):
            pairs.append((query, answer))

    if len(subjects) < 2:
        raise ValueError("At least two TRAIN subjects are required")
    if len(relations) < 2:
        raise ValueError("At least two TRAIN relation types are required")
    return pairs, relations, len(subjects)


class RoleQADataset(Dataset):
    def __init__(self, pairs, tokenizer: Tokenizer, context_length: int):
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

            seq = prompt_ids + answer_ids
            answer_start = len(prompt_ids)
            x = seq[:-1]
            y = seq[1:]
            mask = [1.0 if (i + 1) >= answer_start else 0.0 for i in range(len(y))]

            pad = context_length - len(x)
            x += [tokenizer.pad_id] * pad
            y += [tokenizer.pad_id] * pad
            mask += [0.0] * pad

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

    pairs, relations, subject_count = load_augmented_pairs(Path(args.train))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(args.base_model, device=device)

    freeze_blocks = max(0, min(args.freeze_blocks, len(model.blocks)))
    for index in range(freeze_blocks):
        for parameter in model.blocks[index].parameters():
            parameter.requires_grad = False

    dataset = RoleQADataset(pairs, tokenizer, model.context_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("=" * 112)
    print(" LLM_TRY v10.12.10 Semantic Role Generalization Training")
    print("=" * 112)
    print("Device             :", device)
    if device.type == "cuda":
        print("GPU                :", torch.cuda.get_device_name(0))
    print("Base model         :", args.base_model)
    print("TRAIN subjects     :", subject_count)
    print("TRAIN relations    :", sorted(relations))
    print("Augmented rows     :", len(pairs))
    print("Prompts / row      :", 3)
    print("Frozen blocks      :", freeze_blocks)
    print("Learning rate      :", args.learning_rate)
    print("Epoch limit        :", args.epochs)
    print("Output             :", args.output)
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
        print(
            f"epoch={epoch:02d} train={train_loss:.6f} "
            f"time={time.perf_counter()-started:.2f}s"
        )
        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No semantic-role candidate produced")

    model.load_state_dict(best_state)
    model.to(device)

    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    metadata.update({
        "semantic_role_version": "v10.12.10",
        "semantic_role_train": args.train,
        "semantic_role_train_subjects": subject_count,
        "semantic_role_relations": sorted(relations),
        "semantic_role_augmented_rows": len(pairs),
        "semantic_role_learning_rate": args.learning_rate,
        "semantic_role_best_epoch": best_epoch,
        "semantic_role_frozen_blocks": freeze_blocks,
        "semantic_role_base_model": args.base_model,
    })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(
        str(output),
        optimizer=optimizer,
        epoch=best_epoch,
        loss=best_loss,
        metadata=metadata,
    )

    print()
    print("Best epoch         :", best_epoch)
    print("Best train loss    :", f"{best_loss:.6f}")
    print("Saved candidate    :", output)


if __name__ == "__main__":
    main()
