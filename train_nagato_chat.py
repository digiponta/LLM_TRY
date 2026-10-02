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
    p.add_argument("--data", default="data/nagato_canonical_v91.jsonl")
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    p.add_argument("--base-model", default="model/model-gpu-v0.8-chat-clean.pt")
    p.add_argument("--output", default="model/model-llm-try-nagato-chat-v93.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-data", default="data/nagato_chat_val.jsonl")
    p.add_argument("--anchor-data", default="")
    p.add_argument("--expansion-data", default="")
    p.add_argument("--completion-data", default="")
    p.add_argument("--paraphrase-data", default="")
    p.add_argument("--consistency-data", default="")
    p.add_argument("--unknown-data", default="data/nagato_unknown_paraphrase.jsonl")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Base repeat count for ordinary SFT pairs.",
    )
    p.add_argument("--canonical-identity-weight", type=int, default=12)
    p.add_argument("--canonical-persona-weight", type=int, default=6)
    p.add_argument("--canonical-knowledge-weight", type=int, default=5)
    p.add_argument("--canonical-paraphrase-weight", type=int, default=3)
    p.add_argument("--unknown-weight", type=int, default=4)
    p.add_argument(
        "--persona-weight",
        type=int,
        default=3,
        help="Repeat multiplier for persona/style examples.",
    )
    p.add_argument(
        "--anchor-weight",
        type=int,
        default=12,
        help="Repeat multiplier for identity-anchor/short-persona examples.",
    )
    p.add_argument(
        "--expansion-weight",
        type=int,
        default=3,
        help="Repeat multiplier for longer identity-preserving responses.",
    )
    p.add_argument(
        "--completion-weight",
        type=int,
        default=5,
        help="Repeat multiplier for complete knowledge responses to resist premature EOS.",
    )
    p.add_argument(
        "--paraphrase-weight",
        type=int,
        default=3,
        help="Repeat multiplier for paraphrase generalization examples.",
    )
    p.add_argument(
        "--consistency-weight",
        type=int,
        default=3,
        help="Repeat multiplier for concept-group consistency examples.",
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
        default=True,
        help="Fine-tune the LM head with a lower learning rate.",
    )
    p.add_argument(
        "--lm-head-learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate used only for the LM head.",
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


PERSONA_PATTERNS = (
    "あなたは誰",
    "自己紹介",
    "名前",
    "本は好き",
    "読書",
    "人間について",
    "人について",
    "感情",
    "怒って",
    "楽しい",
    "寂しい",
    "好きな場所",
    "好きな時間",
    "学校は",
    "勉強は好き",
    "何をしていますか",
    "今日はどう",
    "元気ですか",
)


IDENTITY_PATTERNS = (
    "あなたは誰", "名前", "自己紹介",
)

KNOWLEDGE_PATTERNS = (
    "AI", "人工知能", "LLM", "大規模言語モデル",
    "コンピュータ", "GPU", "CUDA", "量子力学", "量子コンピュータ",
    "Semantic", "セマンティック",
)

PARAPHRASE_PATTERNS = (
    "って何", "について教えて", "簡単に説明", "知りたい",
)

def canonical_weight_for_pair(user: str, answer: str, args) -> int:
    text = user.strip()

    if any(p in text for p in IDENTITY_PATTERNS):
        return max(1, args.canonical_identity_weight)

    if is_persona_pair(text, answer):
        return max(1, args.canonical_persona_weight)

    if any(p in text for p in KNOWLEDGE_PATTERNS):
        if any(p in text for p in PARAPHRASE_PATTERNS):
            return max(1, args.canonical_paraphrase_weight)
        return max(1, args.canonical_knowledge_weight)

    return max(1, args.repeat)


def is_persona_pair(user: str, answer: str) -> bool:
    text = user.strip()
    if any(pattern in text for pattern in PERSONA_PATTERNS):
        return True

    # Identity marker in the target answer is always persona-relevant.
    if "長門有希" in answer:
        return True

    return False


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
    val_data_path = Path(args.val_data)
    anchor_data_path = Path(args.anchor_data) if args.anchor_data else None
    expansion_data_path = Path(args.expansion_data) if args.expansion_data else None
    completion_data_path = Path(args.completion_data) if args.completion_data else None
    paraphrase_data_path = Path(args.paraphrase_data) if args.paraphrase_data else None
    consistency_data_path = Path(args.consistency_data) if args.consistency_data else None
    unknown_data_path = Path(args.unknown_data) if args.unknown_data else None
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
    val_pairs = load_pairs(val_data_path)
    anchor_pairs = load_pairs(anchor_data_path) if anchor_data_path else []
    expansion_pairs = load_pairs(expansion_data_path) if expansion_data_path else []
    completion_pairs = load_pairs(completion_data_path) if completion_data_path else []
    paraphrase_pairs = load_pairs(paraphrase_data_path) if paraphrase_data_path else []
    consistency_pairs = load_pairs(consistency_data_path) if consistency_data_path else []
    unknown_pairs = load_pairs(unknown_data_path) if unknown_data_path else []
    train_pairs = list(pairs)
    random.shuffle(train_pairs)

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

    repeated_train: List[Tuple[str, str]] = []
    persona_pairs = 0
    normal_pairs = 0

    canonical_mode = (
        data_path.name == "nagato_canonical_v91.jsonl"
        and not anchor_pairs
        and not expansion_pairs
        and not completion_pairs
        and not paraphrase_pairs
        and not consistency_pairs
    )

    if canonical_mode:
        for user, answer in train_pairs:
            copies = canonical_weight_for_pair(user, answer, args)
            repeated_train.extend([(user, answer)] * copies)
            if is_persona_pair(user, answer):
                persona_pairs += 1
            else:
                normal_pairs += 1
    else:
        for user, answer in train_pairs:
            if is_persona_pair(user, answer):
                persona_pairs += 1
                copies = max(1, args.repeat) * max(1, args.persona_weight)
            else:
                normal_pairs += 1
                copies = max(1, args.repeat)
            repeated_train.extend([(user, answer)] * copies)

        for user, answer in anchor_pairs:
            repeated_train.extend(
                [(user, answer)] * max(1, args.anchor_weight)
            )

        for user, answer in expansion_pairs:
            repeated_train.extend(
                [(user, answer)] * max(1, args.expansion_weight)
            )

        for user, answer in completion_pairs:
            repeated_train.extend(
                [(user, answer)] * max(1, args.completion_weight)
            )

        for user, answer in paraphrase_pairs:
            repeated_train.extend(
                [(user, answer)] * max(1, args.paraphrase_weight)
            )

        for user, answer in consistency_pairs:
            repeated_train.extend(
                [(user, answer)] * max(1, args.consistency_weight)
            )

    for user, answer in unknown_pairs:
        repeated_train.extend(
            [(user, answer)] * max(1, args.unknown_weight)
        )

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

    main_params = []
    lm_head_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("lm_head."):
            lm_head_params.append(parameter)
        else:
            main_params.append(parameter)

    param_groups = []
    if main_params:
        param_groups.append({
            "params": main_params,
            "lr": args.learning_rate,
        })
    if lm_head_params:
        param_groups.append({
            "params": lm_head_params,
            "lr": args.lm_head_learning_rate,
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
    )

    print("=" * 76)
    print(" LLM_TRY Nagato Chat SFT")
    print("=" * 76)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Train data      :", data_path)
    print("Validation data :", val_data_path)
    print("Anchor data     :", anchor_data_path)
    print("Expansion data  :", expansion_data_path)
    print("Completion data :", completion_data_path)
    print("Paraphrase data :", paraphrase_data_path)
    print("Consistency data:", consistency_data_path)
    print("Unknown data    :", unknown_data_path)
    print("Unique train    :", len(pairs))
    print("Train pairs     :", len(train_pairs))
    print("Validation pairs:", len(val_pairs), "(independent paraphrases)")
    print("Anchor pairs    :", len(anchor_pairs))
    print("Expansion pairs :", len(expansion_pairs))
    print("Completion pairs:", len(completion_pairs))
    print("Paraphrase pairs:", len(paraphrase_pairs))
    print("Consistency pairs:", len(consistency_pairs))
    print("Unknown pairs   :", len(unknown_pairs))
    print("Base repeat     :", args.repeat)
    print("Persona weight  :", args.persona_weight)
    print("Anchor weight   :", args.anchor_weight)
    print("Expansion weight:", args.expansion_weight)
    print("Completion weight:", args.completion_weight)
    print("Paraphrase weight:", args.paraphrase_weight)
    print("Consistency weight:", args.consistency_weight)
    print("Unknown weight  :", args.unknown_weight)
    print("Persona pairs   :", persona_pairs)
    print("Normal pairs    :", normal_pairs)
    print("Canonical mode   :", canonical_mode)
    if canonical_mode:
        print("Canonical weights: identity=%d persona=%d knowledge=%d paraphrase=%d general=%d" % (
            args.canonical_identity_weight,
            args.canonical_persona_weight,
            args.canonical_knowledge_weight,
            args.canonical_paraphrase_weight,
            args.repeat,
        ))
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
    print("Block/Norm LR   :", args.learning_rate)
    print("LM head LR      :", args.lm_head_learning_rate)
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
