#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.14 Daily Conversation Runtime regression."""

from chat import classify_daily_conversation, daily_chat_fallback


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main() -> None:
    print("=" * 104)
    print(" LLM_TRY v10.14 Daily Conversation Runtime Regression")
    print("=" * 104)

    casual = {
        "こんにちは": "greeting",
        "こんばんは": "greeting",
        "おはよう": "greeting",
        "今日は疲れた": "feeling",
        "眠い": "feeling",
        "なるほど": "acknowledgement",
        "ありがとう": "acknowledgement",
        "今日はどうしようかな": "casual",
        "あなたは誰？": "persona",
        "長門有希": "persona",
        "元気": "casual",
    }
    for text, expected in casual.items():
        actual = classify_daily_conversation(text)
        check(f"casual:{text}", actual == expected, f"{actual!r}")

    knowledge = (
        "CPUとは",
        "宇宙とは",
        "暗号について教えて",
        "GPUとCPUの違い",
        "なぜGPUは高速なの",
        "Pythonの使い方",
        "宇宙",
        "CPU",
        "CPUは",
    )
    for text in knowledge:
        actual = classify_daily_conversation(text)
        check(f"knowledge-preserved:{text}", actual == "", f"{actual!r}")

    check(
        "fallback:greeting",
        daily_chat_fallback("greeting", "こんにちは") == "こんにちは。",
    )
    check(
        "fallback:feeling",
        "休" in daily_chat_fallback("feeling", "今日は疲れた"),
    )
    check(
        "fallback:persona",
        daily_chat_fallback("persona", "あなたは誰？") == "長門有希。",
    )

    print()
    print("STATUS : DAILY_CONVERSATION_RUNTIME_PASS")


if __name__ == "__main__":
    main()
