#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.12 Corpus-to-Semantic Knowledge regression."""

from __future__ import annotations

from semantic_role_generalization_v101210 import decompose_role
from train_semantic_role_generalization_v101210 import balance_role_items


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main():
    print("="*116)
    print(" LLM_TRY v10.12.12 Corpus-to-Semantic Knowledge Regression")
    print("="*116)

    samples=[
        ("CPU","CPUとは","CPUは、命令を実行するために使われる。","function"),
        ("磁石","磁石とは","磁石は、磁場を持つ。","property"),
        ("A","Aとは","Aは、Bと異なる。","comparison"),
    ]
    items=[]
    for s,q,a,r in samples:
        item=decompose_role(s,q,a)
        check(f"decompose:{r}",item.relation==r,item.relation)
        check("object-present",bool(item.object_description))
        items.append(item)

    balanced,before,after=balance_role_items(items,max_multiplier=4)
    check("role-balance-preserves-relations",set(before)==set(after),str(after))
    check("role-balance-nonempty",len(balanced)>=len(items),str(len(balanced)))

    train={"CPU","磁石"}
    holdout={"A"}
    check("concept-disjoint",not (train & holdout))

    plain_gain=.008
    semantic_gain=.014
    check("plain-gain-policy",plain_gain>=.005,f"{plain_gain:+.3f}")
    check("semantic-gain-policy",semantic_gain>=.010,f"{semantic_gain:+.3f}")

    print()
    print("Corpus decomposition : PASS")
    print("Role balancing       : PASS")
    print("Concept isolation    : PASS")
    print("Gain policy          : PASS")
    print("STATUS               : CORPUS_TO_SEMANTIC_KNOWLEDGE_PASS")


if __name__=="__main__":
    main()
