# train_nagato_full.py
#
# LLM_TRY full-corpus continued pretraining for data-nagato.txt.
#
# Goal:
#   Use 100% of data-nagato.txt for training on every epoch.
#
# Notes:
#   - No validation split is used.
#   - The final tail is included by adding an overlapping last window.
#   - Therefore every next-token target in the corpus is covered.
#   - The input checkpoint is never overwritten unless --output explicitly points to it.

from __future__ import annotations

import argparse
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


SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train LLM_TRY on 100% of data-nagato.txt every epoch."
    )
    p.add_argument("--data", default="data/data-nagato.txt")
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    p.add_argument(
        "--base-model",
        default="model/model-llm-try-nagato-chat-v94.pt",
        help="Checkpoint to continue from.",
    )
    p.add_argument(
        "--output",
        default="model/model-llm-try-nagato-full.pt",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Tokens per training input; clipped to model context length.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Window stride. <= block-size. Tail is always included.",
    )
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Also save epoch checkpoints every N epochs (0 disables).",
    )
    return p.parse_args()


def resolve_data_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    if value == "data/data-nagato.txt":
        fallback = Path("data-nagato.txt")
        if fallback.exists():
            return fallback
    raise FileNotFoundError(
        f"Training data not found: {path}\n"
        "Place data-nagato.txt at data/data-nagato.txt or project root, "
        "or specify --data PATH."
    )


class FullCorpusDataset(Dataset):
    def __init__(
        self,
        token_ids: Sequence[int],
        block_size: int,
        stride: int,
    ):
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if stride < 1 or stride > block_size:
            raise ValueError("stride must satisfy 1 <= stride <= block_size")
        if len(token_ids) < block_size + 1:
            raise ValueError(
                f"Not enough tokens ({len(token_ids)}) for block_size={block_size}. "
                "Reduce --block-size."
            )

        last_start = len(token_ids) - (block_size + 1)
        starts = list(range(0, last_start + 1, stride))
        if not starts:
            starts = [0]
        if starts[-1] != last_start:
            starts.append(last_start)

        self.rows: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.starts = starts
        for start in starts:
            seq = token_ids[start : start + block_size + 1]
            self.rows.append((
                torch.tensor(seq[:-1], dtype=torch.long),
                torch.tensor(seq[1:], dtype=torch.long),
            ))

        # Coverage is computed over target-token indices 1..N-1.
        covered = [False] * len(token_ids)
        for start in starts:
            end = start + block_size
            for target_index in range(start + 1, end + 1):
                if target_index < len(covered):
                    covered[target_index] = True
        self.covered_targets = sum(covered[1:])
        self.total_targets = len(token_ids) - 1

    @property
    def coverage(self) -> float:
        if self.total_targets <= 0:
            return 0.0
        return self.covered_targets / self.total_targets

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def lm_loss(model: LanguageModel, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )


@torch.no_grad()
def evaluate(model: LanguageModel, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_tokens += int(y.numel())
        batches += 1

    mean_loss = total_loss / max(1, batches)
    accuracy = total_correct / max(1, total_tokens)
    return mean_loss, accuracy


def save_checkpoint(model, optimizer, path: Path, epoch: int, loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(
        str(path),
        optimizer=optimizer,
        epoch=epoch,
        loss=loss,
    )


def main() -> None:
    args = parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = resolve_data_path(args.data)
    tokenizer_path = Path(args.tokenizer)
    base_path = Path(args.base_model)
    output_path = Path(args.output)

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    if not base_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_path}")
    if output_path.resolve() == base_path.resolve():
        raise ValueError("--output must differ from --base-model")

    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, checkpoint = LanguageModel.load_checkpoint(str(base_path), device=device)

    if tokenizer.vocab_size != model.vocab_size:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"{tokenizer.vocab_size} != {model.vocab_size}"
        )

    block_size = min(args.block_size, model.context_length)
    stride = min(args.stride, block_size)

    text = data_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Training data is empty: {data_path}")

    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    dataset = FullCorpusDataset(token_ids, block_size=block_size, stride=stride)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    before_loss, before_acc = evaluate(model, eval_loader, device)

    print("=" * 84)
    print(" LLM_TRY Full data-nagato.txt Training")
    print("=" * 84)
    print("Device            :", device)
    if device.type == "cuda":
        print("GPU               :", torch.cuda.get_device_name(0))
    print("Data              :", data_path)
    print("Characters        :", len(text))
    print("Total tokens      :", len(token_ids))
    print("Target tokens     :", dataset.total_targets)
    print("Covered targets   :", dataset.covered_targets)
    print("Corpus coverage   :", f"{dataset.coverage * 100:.2f}%")
    print("Tokenizer         :", tokenizer_path)
    print("Vocabulary        :", tokenizer.vocab_size)
    print("Base model        :", base_path)
    print("Base checkpoint   :", checkpoint.get("loss"))
    print("Parameters        :", f"{model.parameter_count:,}")
    print("Context length    :", model.context_length)
    print("Block size        :", block_size)
    print("Stride            :", stride)
    print("Training windows  :", len(dataset))
    print("Batch size        :", args.batch_size)
    print("Learning rate     :", args.learning_rate)
    print("Epochs            :", args.epochs)
    print("Before corpus loss:", f"{before_loss:.6f}")
    print("Before perplexity :", f"{math.exp(min(before_loss, 20.0)):.3f}")
    print("Before token acc. :", f"{before_acc * 100:.2f}%")
    print()

    final_loss = before_loss

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        total_loss = 0.0
        batches = 0
        total_steps = len(loader)

        for step, (x, y) in enumerate(loader, 1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = lm_loss(model, x, y)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            total_loss += float(loss.item())
            batches += 1

            if step == 1 or step % 10 == 0 or step == total_steps:
                elapsed = time.perf_counter() - started
                rate = elapsed / max(1, step)
                eta = rate * max(0, total_steps - step)
                print(
                    f"epoch={epoch:02d}/{args.epochs:02d} "
                    f"step={step:04d}/{total_steps:04d} "
                    f"loss={loss.item():.6f} "
                    f"eta={eta:.1f}s"
                )

        train_loss = total_loss / max(1, batches)
        corpus_loss, token_acc = evaluate(model, eval_loader, device)
        final_loss = corpus_loss
        elapsed = time.perf_counter() - started

        print(
            f"[epoch {epoch:02d}] "
            f"train={train_loss:.6f} "
            f"corpus={corpus_loss:.6f} "
            f"ppl={math.exp(min(corpus_loss, 20.0)):.3f} "
            f"token_acc={token_acc * 100:.2f}% "
            f"time={elapsed:.1f}s"
        )

        if args.save_every > 0 and epoch % args.save_every == 0:
            epoch_path = output_path.with_name(
                f"{output_path.stem}-epoch{epoch:02d}{output_path.suffix}"
            )
            save_checkpoint(model, optimizer, epoch_path, epoch, corpus_loss)
            print("Saved epoch checkpoint:", epoch_path)

    save_checkpoint(model, optimizer, output_path, args.epochs, final_loss)

    after_loss, after_acc = evaluate(model, eval_loader, device)

    print()
    print("=" * 84)
    print(" Training completed")
    print("=" * 84)
    print("Corpus coverage   :", f"{dataset.coverage * 100:.2f}%")
    print("Before loss       :", f"{before_loss:.6f}")
    print("After loss        :", f"{after_loss:.6f}")
    print("Before perplexity :", f"{math.exp(min(before_loss, 20.0)):.3f}")
    print("After perplexity  :", f"{math.exp(min(after_loss, 20.0)):.3f}")
    print("Before token acc. :", f"{before_acc * 100:.2f}%")
    print("After token acc.  :", f"{after_acc * 100:.2f}%")
    print("Saved checkpoint  :", output_path)
    print()
    print("Coverage check:")
    print(
        "  python eval_nagato_full_coverage.py "
        f"--model {output_path} --data {data_path}"
    )
    print()
    print("Interactive check:")
    print(
        "  python chat.py "
        f"--model {output_path} --tokenizer {tokenizer_path}"
    )


if __name__ == "__main__":
    main()
