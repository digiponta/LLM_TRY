# train_nagato_knowledge_sft_v1086.py
#
# Train on semantic-merged corpus knowledge.
#
# Starts from the stable Nagato chat checkpoint and trains all paraphrases
# for each concept against the SAME merged semantic answer.

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

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
        description="Train v10.8.6 semantic-merged Knowledge SFT."
    )
    p.add_argument(
        "--data",
        default="data/nagato_semantic_merge_v1086.jsonl",
    )
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    p.add_argument(
        "--base-model",
        default="model/model-llm-try-nagato-full-chat.pt",
    )
    p.add_argument(
        "--output",
        default="model/model-llm-try-nagato-full-chat-v1086.pt",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=3e-6)
    p.add_argument("--lm-head-learning-rate", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--repeat", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=0.5)
    return p.parse_args()


def load_pairs(path: Path):
    rows = []
    seen = set()
    if not path.exists():
        raise FileNotFoundError(f"Knowledge SFT data not found: {path}")
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
        q = str(row.get("user", "")).strip()
        a = str(row.get("assistant", "")).strip()
        if not q or not a:
            continue
        key = (q, a)
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)
    if not rows:
        raise ValueError("No valid SFT pairs found.")
    return rows


class ConversationDataset(Dataset):
    def __init__(self, pairs, tokenizer, context_length):
        self.rows = []
        for q, a in pairs:
            prompt_ids = tokenizer.encode(
                f"{USER_PREFIX}{q}\n{AI_PREFIX}",
                add_bos=True,
            )
            answer_ids = tokenizer.encode(a, add_eos=True)
            max_sequence = context_length + 1

            if len(prompt_ids) + len(answer_ids) > max_sequence:
                keep_prompt = max(1, max_sequence - len(answer_ids))
                prompt_ids = prompt_ids[-keep_prompt:]

            if len(prompt_ids) + len(answer_ids) > max_sequence:
                room = max_sequence - len(prompt_ids)
                answer_ids = answer_ids[:max(1, room)]
                if answer_ids:
                    answer_ids[-1] = tokenizer.eos_id

            seq = prompt_ids + answer_ids
            answer_start = len(prompt_ids)
            x = seq[:-1]
            y = seq[1:]
            mask = [
                1.0 if (i + 1) >= answer_start else 0.0
                for i in range(len(y))
            ]

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

    def __getitem__(self, i):
        return self.rows[i]


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(
        args.base_model,
        device=device,
    )

    pairs = load_pairs(Path(args.data))
    train_pairs = pairs * max(1, args.repeat)
    random.shuffle(train_pairs)

    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks[-2:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.final_norm.parameters():
        p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True

    main_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("lm_head."):
            head_params.append(p)
        else:
            main_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": main_params, "lr": args.learning_rate},
            {"params": head_params, "lr": args.lm_head_learning_rate},
        ],
        weight_decay=0.01,
    )

    ds = ConversationDataset(
        train_pairs,
        tokenizer,
        model.context_length,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    print("=" * 92)
    print(" LLM_TRY v10.8.6 Semantic-Merged Knowledge SFT")
    print("=" * 92)
    print("Device        :", device)
    if device.type == "cuda":
        print("GPU           :", torch.cuda.get_device_name(0))
    print("Data          :", args.data)
    print("Unique pairs  :", len(pairs))
    print("Repeat        :", args.repeat)
    print("Training rows :", len(train_pairs))
    print("Base model    :", args.base_model)
    print("Base loss     :", checkpoint.get("loss"))
    print("Output        :", args.output)
    print("Epochs        :", args.epochs)
    print()

    final_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model, x, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip,
            )
            optimizer.step()

            total += float(loss.item())
            steps += 1

        final_loss = total / max(1, steps)
        print(f"epoch={epoch:02d} loss={final_loss:.6f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(
        args.output,
        optimizer=optimizer,
        epoch=args.epochs,
        loss=final_loss,
    )

    print()
    print("Saved:", args.output)
    print("Test:")
    print(f"  python chat.py --model {args.output}")


if __name__ == "__main__":
    main()
