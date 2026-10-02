# eval_nagato_full_coverage.py
#
# Evaluate how well an LLM_TRY checkpoint predicts the complete
# data-nagato.txt corpus.
#
# This is a raw-corpus memorization/coverage diagnostic.  It is not a
# conversational QA benchmark and should be interpreted separately from
# eval_integrated_gate_v97.py and other chat/gate regressions.

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer_bpe import Tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate full-corpus next-token coverage for data-nagato.txt."
    )
    p.add_argument("--data", default="data/data-nagato.txt")
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    p.add_argument("--model", default="model/model-llm-try-nagato-full.pt")
    p.add_argument("--block-size", type=int, default=512)
    return p.parse_args()


def resolve_data_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    if value == "data/data-nagato.txt":
        fallback = Path("data-nagato.txt")
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Training data not found: {path}")


@torch.no_grad()
def evaluate_exact_once(
    model: LanguageModel,
    token_ids: Sequence[int],
    block_size: int,
    device: torch.device,
):
    """Score each corpus target token exactly once.

    The corpus is split into consecutive chunks.  Each chunk receives up to
    block_size preceding input tokens, so no target is duplicated in metrics.
    """
    model.eval()

    total_nll = 0.0
    total_correct = 0
    total_targets = 0

    # target positions are 1 .. len(token_ids)-1
    target_start = 1
    while target_start < len(token_ids):
        target_end = min(len(token_ids), target_start + block_size)

        # Inputs for targets [target_start, target_end)
        x_ids = token_ids[target_start - 1 : target_end - 1]
        y_ids = token_ids[target_start : target_end]

        x = torch.tensor([x_ids], dtype=torch.long, device=device)
        y = torch.tensor([y_ids], dtype=torch.long, device=device)

        logits = model(x)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )

        total_nll += float(nll.item())
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_targets += len(y_ids)
        target_start = target_end

    mean_loss = total_nll / max(1, total_targets)
    accuracy = total_correct / max(1, total_targets)
    return mean_loss, accuracy, total_targets


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = resolve_data_path(args.data)
    tokenizer_path = Path(args.tokenizer)
    model_path = Path(args.model)

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, checkpoint = LanguageModel.load_checkpoint(str(model_path), device=device)

    if tokenizer.vocab_size != model.vocab_size:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"{tokenizer.vocab_size} != {model.vocab_size}"
        )

    text = data_path.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    block_size = min(args.block_size, model.context_length)

    loss, accuracy, scored = evaluate_exact_once(
        model,
        token_ids,
        block_size,
        device,
    )

    print("=" * 84)
    print(" LLM_TRY Full Nagato Corpus Coverage Evaluation")
    print("=" * 84)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Data            :", data_path)
    print("Model           :", model_path)
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Characters      :", len(text))
    print("Total tokens    :", len(token_ids))
    print("Scored targets  :", scored)
    print("Target coverage :", f"{100.0 * scored / max(1, len(token_ids) - 1):.2f}%")
    print("Block size      :", block_size)
    print("Corpus loss     :", f"{loss:.6f}")
    print("Perplexity      :", f"{math.exp(min(loss, 20.0)):.3f}")
    print("Next-token acc. :", f"{accuracy * 100:.2f}%")
    print()
    print("Interpretation:")
    print("  - Target coverage should be 100.00%.")
    print("  - Lower corpus loss/perplexity means the checkpoint models this")
    print("    corpus more strongly.")
    print("  - Next-token accuracy is a strict memorization-oriented diagnostic,")
    print("    not a general conversational accuracy score.")


if __name__ == "__main__":
    main()
