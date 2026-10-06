#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.14.1 Multi-Probe Two-Pass Function Resolver CLI."""

from __future__ import annotations

import argparse
import torch

from model import LanguageModel
from tokenizer_bpe import Tokenizer
from two_pass_function_resolver_v101214 import resolve_two_pass_function
from chat import build_prompt, generate_reply, malformed_or_unstable


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True)
    p.add_argument("--model", default="model/model-gpu-v1.6.2-online.pt")
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(args.model, device=device)

    def generate(query):
        prompt, _ = build_prompt(history=[], user_text=query, history_turns=0)
        return generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=96,
            temperature=0.0,
            top_k=20,
            repetition_penalty=1.05,
            seed=0,
        ).text

    result = resolve_two_pass_function(
        args.subject,
        generate=generate,
        malformed=malformed_or_unstable,
    )

    print("=" * 96)
    print(" LLM_TRY v10.12.14.1 Multi-Probe Two-Pass Function Semantic Resolver")
    print("=" * 96)
    print("Device   :", device)
    if device.type == "cuda":
        print("GPU      :", torch.cuda.get_device_name(0))
    print("Model    :", args.model)
    print("Subject  :", result.subject)
    print()
    print("[Pass 1: generated evidence probes]")
    for i, (query, item) in enumerate(zip(result.evidence_queries, result.evidence_items), 1):
        print(f"{i}. Q: {query}")
        print(f"   A: {item}")
    print()
    print("[Merged evidence]")
    print(result.evidence)
    print()
    print("[Extracted function structure]")
    print("action  :", result.structure.action)
    print("target  :", result.structure.target)
    print("purpose :", result.structure.purpose)
    print()
    print("[Pass 2: function answer]")
    print(result.answer)
    print()
    print("Fallback :", result.used_fallback)


if __name__ == "__main__":
    main()
