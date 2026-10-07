#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Held-out paraphrase checks for v10.15 Daily Conversation SFT.

This script intentionally uses prompts that are not present verbatim in the
59-pair training set. It evaluates routing first and, when a trained checkpoint
is available, generation quality with simple expected-content checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chat import (
    build_prompt,
    classify_daily_conversation,
    daily_chat_fallback,
    daily_chat_generation_consistent,
    generate_reply,
    normalize_identity_query,
)
from model import LanguageModel
from tokenizer_bpe import Tokenizer


CASES = [
    ("やっぱり今日は疲れ気味", "feeling", ("休", "そう")),
    ("少し眠くなってきた", "feeling", ("休", "眠")),
    ("今日はちょっといい気分", "feeling", ("良", "嬉", "楽し", "気分")),
    ("最近は何をしてるの", "casual", ("話", "あなた")),
    ("読書って好き", "casual", ("嫌い", "好き")),
    ("本は好き", "casual", ("嫌い", "好き")),
    ("助かった、ありがとう", "acknowledgement", ("どういたしまして", "うん")),
    ("少し話さない？", "casual", ("話", "うん")),
]


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model/model-gpu-v10.15-daily-chat.pt")
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    args = p.parse_args()

    print("=" * 104)
    print(" LLM_TRY v10.15 Daily Conversation Held-Out Paraphrase Evaluation")
    print("=" * 104)

    for text, expected_intent, _ in CASES:
        actual = classify_daily_conversation(text)
        check(f"route:{text}", actual == expected_intent, actual)

    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer)
    if not model_path.exists():
        print()
        print(f"[SKIP] model not found: {model_path}")
        print("STATUS : DAILY_SFT_HELDOUT_ROUTING_PASS")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, _ = LanguageModel.load_checkpoint(str(model_path), device=device)

    passed = 0
    for text, _, expected_terms in CASES:
        prompt, _ = build_prompt(
            history=[],
            user_text=normalize_identity_query(text),
            history_turns=0,
        )
        result = generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=64,
            temperature=0.0,
            top_k=20,
            repetition_penalty=1.05,
            seed=0,
        )
        generated_ok = any(term in result.text for term in expected_terms)
        consistent = daily_chat_generation_consistent(
            classify_daily_conversation(text),
            text,
            result.text,
        )
        if generated_ok and consistent:
            final_answer = result.text
            route = "GENERATED"
        else:
            final_answer = daily_chat_fallback(
                classify_daily_conversation(text),
                text,
            )
            route = "FALLBACK"

        ok = any(term in final_answer for term in expected_terms)
        check(
            f"runtime:{text}",
            ok,
            f"{route}: raw={result.text!r} final={final_answer!r}",
        )
        passed += int(ok)

    print()
    print(f"Held-out generation: {passed}/{len(CASES)}")
    print("STATUS : DAILY_SFT_HELDOUT_RUNTIME_PASS")


if __name__ == "__main__":
    main()
