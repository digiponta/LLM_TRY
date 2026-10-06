#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.15.1 Balanced Subject-to-Proposition + Semantic Training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import LanguageModel
from tokenizer_bpe import Tokenizer
from batch_repair_preservation_v10121 import protected_internalized_records
from chat import checkpoint_trained_fingerprints
from semantic_role_generalization_v101210 import training_queries
from train_semantic_role_generalization_v101210 import (
    RoleQADataset,
    balance_role_items,
    load_role_items,
    masked_loss,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/nagato_corpus_semantic_train_v101212.jsonl")
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    p.add_argument("--base-model", default="model/model-gpu-v1.6.2-online.pt")
    p.add_argument("--output", default="model/model-gpu-v1.6.2-online-corpus-semantic-candidate.pt")
    p.add_argument("--learning-log", default="data/chat_history.jsonl")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--freeze-blocks", type=int, default=2)
    p.add_argument("--preservation-weight", type=int, default=2)
    p.add_argument("--subject-mapping-weight", type=int, default=1)
    p.add_argument("--max-role-multiplier", type=int, default=4)
    p.add_argument("--grad-clip", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(args.base_model, device=device)

    raw_items = load_role_items(Path(args.train))
    balanced, counts_before, counts_after = balance_role_items(
        raw_items, max_multiplier=args.max_role_multiplier
    )
    pairs = []
    for item in balanced:
        for q in training_queries(item):
            pairs.append((q, item.answer))

    # v10.12.15 formal Subject -> Full Proposition training.
    # The training input explicitly encodes:
    #     subject => subject + predicate/full proposition
    # while the target remains the canonical full proposition.
    subject_mapping_rows = 0
    subject_to_answer = {item.subject: item.answer for item in raw_items}
    for subject, answer in subject_to_answer.items():
        # Formal lookup form:
        #     subject => [generate full proposition]
        # Do NOT feed the full right-hand proposition back as input; that would
        # duplicate the target and over-weight copying rather than retrieval.
        mapping_query = f"{subject} =>"
        for _ in range(max(1, args.subject_mapping_weight)):
            pairs.append((mapping_query, answer))
            subject_mapping_rows += 1

    protected = protected_internalized_records(
        Path(args.learning_log),
        set(checkpoint_trained_fingerprints(checkpoint)),
        set(),
        set(),
    )
    for record in protected:
        for _ in range(max(0, args.preservation_weight)):
            pairs.append((record.question, record.teacher_answer))

    freeze_blocks = max(0, min(args.freeze_blocks, len(model.blocks)))
    for i in range(freeze_blocks):
        for p in model.blocks[i].parameters():
            p.requires_grad = False

    dataset = RoleQADataset(pairs, tokenizer, model.context_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)

    print("=" * 116)
    print(" LLM_TRY v10.12.15.1 Balanced Subject-to-Proposition + Semantic Training")
    print("=" * 116)
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Source propositions :", len(raw_items))
    print("Role counts before  :", counts_before)
    print("Role counts after   :", counts_after)
    print("Protected concepts  :", len(protected))
    print("Subject map weight  :", args.subject_mapping_weight)
    print("Subject map rows    :", subject_mapping_rows)
    print("Total train rows    :", len(pairs))

    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model, x, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            total += float(loss.item())
            batches += 1
        mean = total / max(1, batches)
        print(f"epoch={epoch:02d} train={mean:.6f}")
        if mean < best_loss:
            best_loss = mean
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No candidate produced")

    model.load_state_dict(best_state)
    metadata = checkpoint.get("metadata", {})
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    metadata.update({
        "corpus_semantic_version": "v10.12.15.1",
        "subject_to_proposition_version": "v10.12.15.1",
        "subject_mapping_weight": args.subject_mapping_weight,
        "subject_mapping_rows": subject_mapping_rows,
        "corpus_semantic_train": args.train,
        "corpus_semantic_source_propositions": len(raw_items),
        "corpus_semantic_role_counts_before": counts_before,
        "corpus_semantic_role_counts_after": counts_after,
        "corpus_semantic_preservation_weight": args.preservation_weight,
        "corpus_semantic_best_epoch": best_epoch,
    })
    model.save_checkpoint(
        args.output,
        optimizer=optimizer,
        epoch=best_epoch,
        loss=best_loss,
        metadata=metadata,
    )
    print("Best epoch          :", best_epoch)
    print("Best train loss     :", f"{best_loss:.6f}")
    print("Saved candidate     :", args.output)


if __name__ == "__main__":
    main()
