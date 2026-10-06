# train_nagato_knowledge_sft_v1084.py
#
# Fine-tune an existing Nagato chat checkpoint on corpus-derived knowledge QA.
#
# Conservative design:
#   - final 2 transformer blocks + final norm
#   - LM head at a lower learning rate
#   - low LR
#   - small epoch count
#   - original checkpoint is not overwritten by default

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

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
        description="Corpus-derived Knowledge SFT for LLM_TRY v10.8.4."
    )
    p.add_argument("--data", default="data/nagato_corpus_knowledge_v1084.jsonl")
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    p.add_argument(
        "--base-model",
        default="model/model-llm-try-nagato-full-chat.pt",
    )
    p.add_argument(
        "--output",
        default="model/model-llm-try-nagato-full-chat-v1084.pt",
    )
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-6)
    p.add_argument("--lm-head-learning-rate", type=float, default=5e-7)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=0.5)
    return p.parse_args()


def load_pairs(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Knowledge SFT data not found: {path}")
    pairs = []
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
    if not pairs:
        raise ValueError("No valid SFT pairs found.")
    return pairs


class ConversationDataset(Dataset):
    def __init__(self, pairs, tokenizer, context_length):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.data)
    tokenizer_path = Path(args.tokenizer)
    base_path = Path(args.base_model)
    output_path = Path(args.output)

    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, checkpoint = LanguageModel.load_checkpoint(str(base_path), device=device)
    pairs = load_pairs(data_path)

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

    ds = ConversationDataset(pairs, tokenizer, model.context_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    print("=" * 92)
    print(" LLM_TRY v10.8.4 Corpus-Derived Knowledge SFT")
    print("=" * 92)
    print("Device        :", device)
    if device.type == "cuda":
        print("GPU           :", torch.cuda.get_device_name(0))
    print("Data          :", data_path)
    print("Pairs         :", len(pairs))
    print("Base model    :", base_path)
    print("Base loss     :", checkpoint.get("loss"))
    print("Output        :", output_path)
    print("Epochs        :", args.epochs)
    print("Block/Norm LR :", args.learning_rate)
    print("LM head LR    :", args.lm_head_learning_rate)
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(
        str(output_path),
        optimizer=optimizer,
        epoch=args.epochs,
        loss=final_loss,
    )

    print()
    print("Saved:", output_path)
    print()
    print("Test:")
    print(f"  python chat.py --model {output_path}")


if __name__ == "__main__":
    main()
