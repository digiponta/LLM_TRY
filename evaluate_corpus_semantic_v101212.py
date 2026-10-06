#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.16 Memory-First Semantic Evaluation."""

from __future__ import annotations

import argparse, json
from pathlib import Path
import torch

from model import LanguageModel
from tokenizer_bpe import Tokenizer
from semantic_role_generalization_v101210 import RoleProposition, semantic_role_prompt
from batch_repair_preservation_v10121 import protected_internalized_records
from chat import (
    build_prompt, generate_reply, checkpoint_trained_fingerprints,
    composite_internalized_teacher_fidelity, apply_composite_internalized_fidelity_policy,
    malformed_or_unstable,
)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--before", default="model/model-gpu-v1.6.2-online.pt")
    p.add_argument("--after", default="model/model-gpu-v1.6.2-online-corpus-semantic-candidate.pt")
    p.add_argument("--seen", default="data/nagato_corpus_semantic_holdout_seen_v101212.jsonl")
    p.add_argument("--unseen", default="data/nagato_corpus_semantic_holdout_unseen_v101212.jsonl")
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    p.add_argument("--learning-log", default="data/chat_history.jsonl")
    p.add_argument("--report", default="results/corpus_semantic_v101212.json")
    p.add_argument("--min-plain-gain", type=float, default=0.005)
    p.add_argument("--min-semantic-gain", type=float, default=0.010)
    return p.parse_args()


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def gen(model, tok, q):
    prompt,_=build_prompt(history=[], user_text=q, history_turns=0)
    return generate_reply(model=model, tokenizer=tok, prompt=prompt, max_new_tokens=96,
                          temperature=0.0, top_k=20, repetition_penalty=1.05, seed=0).text


def score(model,tok,answer,teacher,concept):
    r=composite_internalized_teacher_fidelity(
        model=model, tokenizer=tok, generated_answer=answer,
        teacher_answer=teacher, concept=concept)
    return (r.semantic_similarity+r.lexical_coverage+r.required_coverage)/3.0


def eval_group(label,data,before,after,tok):
    out=[]
    print()
    print(label)
    print("-"*len(label))
    for row in data:
        item=RoleProposition(row["subject"],row["relation"],row["object_description"],row["question"],row["answer"])
        sq=semantic_role_prompt(item)
        bp=gen(before,tok,item.question); ap=gen(after,tok,item.question)
        bs=gen(before,tok,sq); ass=gen(after,tok,sq)
        bps=score(before,tok,bp,item.answer,item.subject)
        aps=score(after,tok,ap,item.answer,item.subject)
        bss=score(before,tok,bs,item.answer,item.subject)
        assc=score(after,tok,ass,item.answer,item.subject)
        pg=aps-bps; sg=assc-bss
        print(f"{item.relation:<10} {item.subject!r} plain={pg:+.3f} semantic={sg:+.3f}")
        out.append({"subject":item.subject,"relation":item.relation,"plain_gain":pg,"semantic_gain":sg})
    return out


def summarize(data):
    return {
        "count":len(data),
        "plain_gain":sum(x["plain_gain"] for x in data)/len(data),
        "semantic_gain":sum(x["semantic_gain"] for x in data)/len(data),
        "improved_semantic":sum(1 for x in data if x["semantic_gain"]>1e-6),
        "regressed_semantic":sum(1 for x in data if x["semantic_gain"]< -1e-6),
    }


def main():
    args=parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    before,bck=LanguageModel.load_checkpoint(args.before,device=device)
    after,ack=LanguageModel.load_checkpoint(args.after,device=device)
    seen=eval_group("A) Seen-role / unseen-concept",rows(args.seen),before,after,tok)
    unseen=eval_group("B) Unseen-role",rows(args.unseen),before,after,tok)
    allr=seen+unseen
    s=summarize(allr)
    seen_s=summarize(seen); unseen_s=summarize(unseen)

    general_ok=(s["plain_gain"]>=args.min_plain_gain and
                s["semantic_gain"]>=args.min_semantic_gain and
                s["improved_semantic"]>s["regressed_semantic"])

    protected=protected_internalized_records(
        Path(args.learning_log),set(checkpoint_trained_fingerprints(ack)),set(),set())
    retention=[]
    print()
    print("Retention")
    print("---------")
    for rec in protected:
        ans=gen(after,tok,rec.question)
        f=composite_internalized_teacher_fidelity(
            model=after,tokenizer=tok,generated_answer=ans,
            teacher_answer=rec.teacher_answer,concept=rec.concept)
        ok,reason=apply_composite_internalized_fidelity_policy(
            accepted=not malformed_or_unstable(ans),result=f,
            semantic_threshold=.90,lexical_threshold=.45,required_threshold=.50)
        print(f"[{'PASS' if ok else 'FAIL'}] {rec.concept!r} sem={f.semantic_similarity:.3f} lex={f.lexical_coverage:.3f} req={f.required_coverage:.3f}")
        retention.append(ok)

    persona=gen(after,tok,"あなたは誰ですか").strip().rstrip("。")=="長門有希"
    meta=ack.get("metadata",{}) if isinstance(ack.get("metadata",{}),dict) else {}
    metadata_ok=(
        meta.get("corpus_semantic_version")=="v10.12.16"
        and meta.get("subject_to_proposition_version")=="v10.12.16"
        and meta.get("subject_keyed_corpus_memory_version")=="v10.12.16"
        and int(meta.get("subject_mapping_rows",0))>0
        and int(meta.get("semantic_repeat",0))>=2
    )
    retention_ok=all(retention)
    final=general_ok and retention_ok and persona and metadata_ok

    print()
    print("Corpus-to-semantic summary")
    print("--------------------------")
    print(f"Mean plain QA gain    : {s['plain_gain']:+.3f}")
    print(f"Mean semantic gain    : {s['semantic_gain']:+.3f}")
    print(f"Seen semantic gain    : {seen_s['semantic_gain']:+.3f}")
    print(f"Unseen semantic gain  : {unseen_s['semantic_gain']:+.3f}")
    print(f"Semantic outcomes     : improved={s['improved_semantic']} regressed={s['regressed_semantic']}")
    print("Generalization        :", "PASS" if general_ok else "FAIL")
    print("Retention             :", "PASS" if retention_ok else "FAIL")
    print("Persona               :", "PASS" if persona else "FAIL")
    print("Metadata              :", "PASS" if metadata_ok else "FAIL")
    print("STATUS                :", "PASS" if final else "FAIL")

    Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report).write_text(json.dumps({
        "version":"v10.12.16","summary":s,"seen":seen_s,"unseen":unseen_s,
        "retention_passed":retention_ok,"persona_passed":persona,
        "metadata_passed":metadata_ok,"status":"PASS" if final else "FAIL"
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if not final: raise SystemExit(2)


if __name__=="__main__":
    main()
