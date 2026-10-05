# train_nagato_full_chat.py
#
# LLM_TRY v10.8 full-Nagato training pipeline.
#
# Correct training order:
#   1) generic chat-clean base
#   2) 100% raw data-nagato.txt continued pretraining
#   3) Nagato canonical conversational SFT
#
# This avoids applying raw LM training after persona/chat SFT, which can
# damage conversational behavior and identity response quality.

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run full Nagato raw training followed by canonical chat SFT."
    )
    p.add_argument("--data", default="data/data-nagato.txt")
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    p.add_argument("--base-model", default="model/model-gpu-v0.8-chat-clean.pt")
    p.add_argument(
        "--raw-output",
        default="model/model-llm-try-nagato-full-raw.pt",
    )
    p.add_argument(
        "--output",
        default="model/model-llm-try-nagato-full-chat.pt",
    )
    p.add_argument("--raw-epochs", type=int, default=5)
    p.add_argument("--raw-learning-rate", type=float, default=1e-5)
    p.add_argument("--raw-batch-size", type=int, default=8)
    p.add_argument("--sft-epochs", type=int, default=10)
    p.add_argument("--sft-learning-rate", type=float, default=5e-6)
    p.add_argument("--sft-lm-head-learning-rate", type=float, default=1e-6)
    p.add_argument("--sft-batch-size", type=int, default=8)
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print()
    print("=" * 88)
    print("RUN:", " ".join(cmd))
    print("=" * 88)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    for required in (
        Path("train_nagato_full.py"),
        Path("train_nagato_chat.py"),
        Path(args.data),
        Path(args.tokenizer),
        Path(args.base_model),
        Path("data/nagato_canonical_v91.jsonl"),
        Path("data/nagato_chat_val.jsonl"),
    ):
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    python = sys.executable

    print("=" * 88)
    print(" LLM_TRY v10.11.5 Full Nagato -> Canonical Chat SFT Pipeline")
    print("=" * 88)
    print("Stage 1 base   :", args.base_model)
    print("Raw corpus     :", args.data)
    print("Stage 1 output :", args.raw_output)
    print("Stage 2 output :", args.output)
    print()
    print("Training order:")
    print("  generic chat-clean")
    print("       -> full raw Nagato corpus")
    print("       -> conflict-resolved canonical Nagato conversational SFT")

    run([
        python,
        "train_nagato_full.py",
        "--data", args.data,
        "--tokenizer", args.tokenizer,
        "--base-model", args.base_model,
        "--output", args.raw_output,
        "--epochs", str(args.raw_epochs),
        "--learning-rate", str(args.raw_learning_rate),
        "--batch-size", str(args.raw_batch_size),
        "--block-size", "512",
        "--stride", "512",
    ])

    # Canonical-only SFT.
    # nagato_canonical_v91.jsonl already resolves conflicting supervision from
    # anchor/expansion/completion/paraphrase/consistency sources. Re-injecting
    # those source files here would reintroduce the conflicts that canonical
    # construction intentionally removed.
    run([
        python,
        "train_nagato_chat.py",
        "--data", "data/nagato_canonical_v91.jsonl",
        "--val-data", "data/nagato_chat_val.jsonl",
        "--tokenizer", args.tokenizer,
        "--base-model", args.raw_output,
        "--output", args.output,
        "--epochs", str(args.sft_epochs),
        "--learning-rate", str(args.sft_learning_rate),
        "--lm-head-learning-rate", str(args.sft_lm_head_learning_rate),
        "--batch-size", str(args.sft_batch_size),
        "--trainable-blocks", "2",
        "--canonical-identity-weight", "12",
        "--canonical-persona-weight", "6",
        "--canonical-knowledge-weight", "5",
        "--canonical-paraphrase-weight", "3",
        "--repeat", "1",
    ])

    # Final identity-generalization stabilization.  These short prompts are
    # intentionally trained last so bare/pronoun identity queries remain
    # reliable after the broader conversational SFT.
    run([
        python,
        "train_nagato_chat.py",
        "--data", "data/nagato_identity_generalization_v1081.jsonl",
        "--val-data", "data/nagato_chat_val.jsonl",
        "--tokenizer", args.tokenizer,
        "--base-model", args.output,
        "--output", args.output,
        "--epochs", "4",
        "--learning-rate", "2e-6",
        "--lm-head-learning-rate", "5e-7",
        "--batch-size", "6",
        "--trainable-blocks", "2",
        "--repeat", "6",
        "--persona-weight", "1",
    ])

    print()
    print("=" * 88)
    print(" Pipeline completed")
    print("=" * 88)
    print("Raw checkpoint :", args.raw_output)
    print("Final checkpoint:", args.output)
    print()
    print("Test:")
    print(f"  {python} chat.py --model {args.output}")
    print()
    print("Raw corpus coverage:")
    print(
        f"  {python} eval_nagato_full_coverage.py "
        f"--model {args.output} --data {args.data}"
    )


if __name__ == "__main__":
    main()
