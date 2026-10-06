# train_nagato.py
#
# LLM_TRY v10.12.4 moderate raw-text continued pretraining for data-nagato.txt.
#
# This script:
#   - reuses the existing byte-level BPE tokenizer,
#   - loads an existing LLM_TRY checkpoint,
#   - trains causal next-token prediction on data-nagato.txt,
#   - keeps a validation split,
#   - saves the best model to a NEW checkpoint.
#
# It does not overwrite the base checkpoint.

from __future__ import annotations

import argparse
import math
import random
import json
import shutil
import time
from datetime import datetime
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
        description="Moderately continue LLM_TRY language-model training on data-nagato.txt while preserving checkpoint metadata."
    )
    p.add_argument(
        "--data",
        default="data/data-nagato.txt",
        help="Training text. If the default path is absent, ./data-nagato.txt is also tried.",
    )
    p.add_argument(
        "--tokenizer",
        default="model/tokenizer-v0.7-bpe.json",
    )
    p.add_argument(
        "--base-model",
        default="model/model-gpu-v1.6.2-online.pt",
    )
    p.add_argument(
        "--output",
        default="model/model-gpu-v1.6.2-online-nagato-candidate.pt",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Tokens per training input. Must be <= model context length.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Token stride between neighboring training windows.",
    )
    p.add_argument("--validation-ratio", type=float, default=0.10)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--freeze-blocks",
        type=int,
        default=2,
        help=(
            "Freeze the first N transformer blocks during moderate corpus "
            "adaptation. Default: 2."
        ),
    )
    p.add_argument(
        "--baseline-dir",
        default="model/baselines",
        help="Directory for timestamped pre-Nagato baseline snapshots.",
    )
    p.add_argument(
        "--baseline-manifest",
        default="model/baselines/nagato_baseline_latest.json",
        help="Manifest pointing to the latest timestamped pre-Nagato baseline.",
    )
    p.add_argument(
        "--baseline-snapshot",
        default="model/model-gpu-v1.6.2-online-pre-nagato.pt",
        help=(
            "Compatibility baseline path. Created only if absent; timestamped "
            "snapshots in --baseline-dir are the authoritative per-run baselines."
        ),
    )
    return p.parse_args()


def resolve_data_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path

    # Convenient fallback for repositories where the file sits at project root.
    if value == "data/data-nagato.txt":
        fallback = Path("data-nagato.txt")
        if fallback.exists():
            return fallback

    raise FileNotFoundError(
        f"Training data not found: {path}\n"
        "Place data-nagato.txt at data/data-nagato.txt or project root, "
        "or specify --data PATH."
    )


class TokenBlockDataset(Dataset):
    def __init__(
        self,
        token_ids: Sequence[int],
        block_size: int,
        stride: int,
    ):
        if block_size < 1:
            raise ValueError("block_size must be >= 1.")
        if stride < 1:
            raise ValueError("stride must be >= 1.")
        if len(token_ids) < block_size + 1:
            raise ValueError(
                f"Not enough tokens ({len(token_ids)}) for block_size={block_size}. "
                "Reduce --block-size."
            )

        self.rows: List[Tuple[torch.Tensor, torch.Tensor]] = []
        last_start = len(token_ids) - (block_size + 1)

        starts = list(range(0, last_start + 1, stride))
        if starts[-1] != last_start:
            starts.append(last_start)

        for start in starts:
            seq = token_ids[start : start + block_size + 1]
            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            self.rows.append((x, y))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def split_tokens(
    token_ids: Sequence[int],
    validation_ratio: float,
    min_tokens_per_split: int,
) -> Tuple[List[int], List[int]]:
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must satisfy 0 <= ratio < 1.")

    ids = list(token_ids)

    if validation_ratio == 0.0:
        return ids, []

    val_count = max(
        min_tokens_per_split,
        int(round(len(ids) * validation_ratio)),
    )

    if len(ids) - val_count < min_tokens_per_split:
        raise ValueError(
            "Dataset is too small for the requested validation split and block size. "
            "Reduce --block-size or use --validation-ratio 0."
        )

    # Keep the validation text contiguous and at the end of the corpus.
    return ids[:-val_count], ids[-val_count:]


def lm_loss(model: LanguageModel, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )


@torch.no_grad()
def evaluate(
    model: LanguageModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        loss = lm_loss(model, x, y)
        total_loss += float(loss.item())
        batches += 1

    return total_loss / max(1, batches)


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

    # v10.12.11.1: create an immutable per-run baseline before every
    # continued-pretraining run. The timestamp includes microseconds so repeated
    # launches within the same second cannot overwrite one another.
    snapshot_dir = Path(args.baseline_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    timestamped_baseline = (
        snapshot_dir
        / f"model-gpu-v1.6.2-online-pre-nagato-{snapshot_stamp}.pt"
    )
    shutil.copy2(base_path, timestamped_baseline)

    manifest_path = Path(args.baseline_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "v10.12.11.1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_model": str(base_path),
        "snapshot": str(timestamped_baseline),
        "candidate_output": str(output_path),
        "data": str(data_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[timestamped pre-Nagato baseline created: {timestamped_baseline}]")
    print(f"[latest baseline manifest updated: {manifest_path}]")

    # Preserve the historical fixed path for scripts/users that still reference
    # it. It is never overwritten automatically.
    baseline_snapshot = Path(args.baseline_snapshot)
    if not baseline_snapshot.exists():
        baseline_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_path, baseline_snapshot)
        print(f"[compatibility baseline created: {baseline_snapshot}]")
    else:
        print(f"[compatibility baseline retained: {baseline_snapshot}]")

    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, checkpoint = LanguageModel.load_checkpoint(
        str(base_path),
        device=device,
    )

    if tokenizer.vocab_size != model.vocab_size:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"{tokenizer.vocab_size} != {model.vocab_size}"
        )

    freeze_blocks = max(0, min(args.freeze_blocks, len(model.blocks)))
    for index in range(freeze_blocks):
        for parameter in model.blocks[index].parameters():
            parameter.requires_grad = False

    block_size = min(args.block_size, model.context_length)
    if block_size != args.block_size:
        print(
            f"[WARN] block-size {args.block_size} exceeds model context "
            f"{model.context_length}; using {block_size}."
        )

    text = data_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Training data is empty: {data_path}")

    token_ids = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True,
    )

    train_ids, val_ids = split_tokens(
        token_ids,
        validation_ratio=args.validation_ratio,
        min_tokens_per_split=block_size + 1,
    )

    train_ds = TokenBlockDataset(
        train_ids,
        block_size=block_size,
        stride=args.stride,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = None
    if val_ids:
        val_ds = TokenBlockDataset(
            val_ids,
            block_size=block_size,
            stride=args.stride,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
        )

    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("=" * 76)
    print(" LLM_TRY v10.12.11.1 Timestamped Baseline + Moderate data-nagato Retraining")
    print("=" * 76)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Data            :", data_path)
    print("Characters      :", len(text))
    print("Total tokens    :", len(token_ids))
    print("Train tokens    :", len(train_ids))
    print("Validation tok. :", len(val_ids))
    print("Tokenizer       :", tokenizer_path)
    print("Vocabulary      :", tokenizer.vocab_size)
    print("Base model      :", base_path)
    print("Base loss       :", checkpoint.get("loss"))
    print("Parameters      :", f"{model.parameter_count:,}")
    print("Context length  :", model.context_length)
    print("Block size      :", block_size)
    print("Stride          :", args.stride)
    print("Train windows   :", len(train_ds))
    print("Validation win. :", 0 if val_loader is None else len(val_loader.dataset))
    print("Batch size      :", args.batch_size)
    print("Learning rate   :", args.learning_rate)
    print("Epoch limit     :", args.epochs)
    print("Frozen blocks   :", freeze_blocks)
    print("Training mode   : moderate / metadata-preserving / candidate-only")
    print()

    best_metric = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        started = time.perf_counter()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = lm_loss(model, x, y)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )

            optimizer.step()

            total_loss += float(loss.item())
            batches += 1

        train_loss = total_loss / max(1, batches)
        elapsed = time.perf_counter() - started

        if val_loader is None:
            metric = train_loss
            print(
                f"epoch={epoch:02d} "
                f"train={train_loss:.6f} "
                f"ppl={math.exp(min(train_loss, 20.0)):.2f} "
                f"time={elapsed:.2f}s"
            )
        else:
            val_loss = evaluate(model, val_loader, device)
            metric = val_loss
            print(
                f"epoch={epoch:02d} "
                f"train={train_loss:.6f} "
                f"val={val_loss:.6f} "
                f"ppl={math.exp(min(val_loss, 20.0)):.2f} "
                f"time={elapsed:.2f}s"
            )

        if metric < best_metric - 1e-5:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if val_loader is not None and bad_epochs >= args.patience:
                print("early stopping")
                break

    if best_state is None:
        raise RuntimeError("No valid checkpoint produced.")

    model.load_state_dict(best_state)
    model.to(device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    inherited_metadata = checkpoint.get("metadata", {})
    if not isinstance(inherited_metadata, dict):
        inherited_metadata = {}
    metadata = dict(inherited_metadata)
    metadata.update({
        "nagato_moderate_version": "v10.12.4",
        "nagato_retraining_version": "v10.12.11.1",
        "nagato_moderate_data": str(data_path),
        "nagato_moderate_epochs": best_epoch,
        "nagato_moderate_learning_rate": args.learning_rate,
        "nagato_moderate_block_size": block_size,
        "nagato_moderate_stride": args.stride,
        "nagato_moderate_validation_ratio": args.validation_ratio,
        "nagato_moderate_frozen_blocks": freeze_blocks,
        "nagato_moderate_base_model": str(base_path),
        "nagato_moderate_baseline_snapshot": str(timestamped_baseline),
        "nagato_moderate_compatibility_baseline": str(baseline_snapshot),
        "nagato_moderate_baseline_manifest": str(manifest_path),
    })

    model.save_checkpoint(
        str(output_path),
        optimizer=optimizer,
        epoch=best_epoch,
        loss=best_metric,
        metadata=metadata,
    )

    print()
    print("Training completed.")
    print("Best epoch      :", best_epoch)
    print(
        "Best metric     :",
        f"{best_metric:.6f}",
        "(validation loss)" if val_loader is not None else "(training loss)",
    )
    print("Saved checkpoint:", output_path)
    print(
        "Metadata binding :",
        len(metadata.get("trained_fingerprints", []))
        if isinstance(metadata.get("trained_fingerprints", []), list)
        else 0,
        "fingerprint(s) preserved",
    )
    print()
    print("To test the trained model:")
    print(
        "  python chat.py "
        f"--model {output_path} "
        f"--tokenizer {tokenizer_path}"
    )


if __name__ == "__main__":
    main()
