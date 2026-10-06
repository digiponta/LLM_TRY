#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.13 Function Semantic Decomposition Evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import LanguageModel
from tokenizer_bpe import Tokenizer
from semantic_role_generalization_v101210 import RoleProposition, semantic_role_prompt
from function_semantic_decomposition_v101213 import function_structure_prompt
from batch_repair_preservation_v10121 import protected_internalized_records
from chat import (
    apply_composite_internalized_fidelity_policy,
    build_prompt,
    checkpoint_trained_fingerprints,
    composite_internalized_teacher_fidelity,
    generate_reply,
    malformed_or_unstable,
)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--before",default="model/model-gpu-v1.6.2-online.pt")
    p.add_argument("--after",default="model/model-gpu-v1.6.2-online-function-semantic-candidate.pt")
    p.add_argument("--seen",default="data/nagato_corpus_semantic_holdout_seen_v101212.jsonl")
    p.add_argument("--unseen",default="data/nagato_corpus_semantic_holdout_unseen_v101212.jsonl")
    p.add_argument("--tokenizer",default="model/tokenizer-v0.7-bpe.json")
    p.add_argument("--learning-log",default="data/chat_history.jsonl")
    p.add_argument("--report",default="results/function_semantic_v101213.json")
    p.add_argument("--min-function-semantic-gain",type=float,default=0.005)
    p.add_argument("--min-function-structured-gain",type=float,default=0.010)
    p.add_argument("--min-nonfunction-guard",type=float,default=-0.005)
    return p.parse_args()


def load_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def gen(model,tok,q):
    prompt,_=build_prompt(history=[],user_text=q,history_turns=0)
    return generate_reply(
        model=model,tokenizer=tok,prompt=prompt,max_new_tokens=96,
        temperature=0.0,top_k=20,repetition_penalty=1.05,seed=0
    ).text


def score(model,tok,answer,teacher,concept):
    r=composite_internalized_teacher_fidelity(
        model=model,tokenizer=tok,generated_answer=answer,
        teacher_answer=teacher,concept=concept
    )
    return (r.semantic_similarity+r.lexical_coverage+r.required_coverage)/3.0


def gain_for_query(before,after,tok,query,teacher,concept):
    b=gen(before,tok,query)
    a=gen(after,tok,query)
    return (
        score(after,tok,a,teacher,concept)
        - score(before,tok,b,teacher,concept)
    )


def main():
    args=parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    before,bck=LanguageModel.load_checkpoint(args.before,device=device)
    after,ack=LanguageModel.load_checkpoint(args.after,device=device)

    all_rows=load_rows(args.seen)+load_rows(args.unseen)
    function_rows=[r for r in all_rows if r["relation"]=="function"]
    nonfunction_rows=[r for r in all_rows if r["relation"]!="function"]
    if not function_rows:
        raise RuntimeError("No function HOLDOUT rows found")

    print("="*116)
    print(" LLM_TRY v10.12.13 Function Semantic Decomposition Evaluation")
    print("="*116)
    print("Device                 :",device)
    if device.type=="cuda":
        print("GPU                    :",torch.cuda.get_device_name(0))
    print("Function HOLDOUT       :",len(function_rows))
    print("Non-function guard rows:",len(nonfunction_rows))

    function_results=[]
    print()
    print("Function HOLDOUT")
    print("----------------")
    for row in function_rows:
        item=RoleProposition(
            row["subject"],row["relation"],row["object_description"],row["question"],row["answer"]
        )
        plain_gain=gain_for_query(before,after,tok,item.question,item.answer,item.subject)
        semantic_gain=gain_for_query(
            before,after,tok,semantic_role_prompt(item),item.answer,item.subject
        )
        structured_gain=gain_for_query(
            before,after,tok,function_structure_prompt(item),item.answer,item.subject
        )
        print(
            f"{item.subject!r} plain={plain_gain:+.3f} "
            f"semantic={semantic_gain:+.3f} structured={structured_gain:+.3f}"
        )
        function_results.append({
            "subject":item.subject,
            "plain_gain":plain_gain,
            "semantic_gain":semantic_gain,
            "structured_gain":structured_gain,
        })

    mean_sem=sum(x["semantic_gain"] for x in function_results)/len(function_results)
    mean_struct=sum(x["structured_gain"] for x in function_results)/len(function_results)
    improved_sem=sum(1 for x in function_results if x["semantic_gain"]>1e-6)
    regressed_sem=sum(1 for x in function_results if x["semantic_gain"]< -1e-6)
    improved_struct=sum(1 for x in function_results if x["structured_gain"]>1e-6)
    regressed_struct=sum(1 for x in function_results if x["structured_gain"]< -1e-6)

    guard_gains=[]
    for row in nonfunction_rows:
        item=RoleProposition(
            row["subject"],row["relation"],row["object_description"],row["question"],row["answer"]
        )
        guard_gains.append(
            gain_for_query(before,after,tok,semantic_role_prompt(item),item.answer,item.subject)
        )
    guard_mean=sum(guard_gains)/len(guard_gains) if guard_gains else 0.0

    function_sem_ok=(
        mean_sem>=args.min_function_semantic_gain
        and improved_sem>regressed_sem
    )
    function_struct_ok=(
        mean_struct>=args.min_function_structured_gain
        and improved_struct>regressed_struct
    )
    guard_ok=guard_mean>=args.min_nonfunction_guard

    protected=protected_internalized_records(
        Path(args.learning_log),
        set(checkpoint_trained_fingerprints(ack)),
        set(),
        set(),
    )
    retention=[]
    print()
    print("Retention")
    print("---------")
    for rec in protected:
        ans=gen(after,tok,rec.question)
        f=composite_internalized_teacher_fidelity(
            model=after,tokenizer=tok,generated_answer=ans,
            teacher_answer=rec.teacher_answer,concept=rec.concept
        )
        ok,reason=apply_composite_internalized_fidelity_policy(
            accepted=not malformed_or_unstable(ans),result=f,
            semantic_threshold=.90,lexical_threshold=.45,required_threshold=.50
        )
        print(
            f"[{'PASS' if ok else 'FAIL'}] {rec.concept!r} "
            f"sem={f.semantic_similarity:.3f} lex={f.lexical_coverage:.3f} req={f.required_coverage:.3f}"
        )
        retention.append(ok)

    persona=gen(after,tok,"あなたは誰ですか").strip().rstrip("。")=="長門有希"
    meta=ack.get("metadata",{}) if isinstance(ack.get("metadata",{}),dict) else {}
    metadata_ok=meta.get("function_semantic_version")=="v10.12.13"
    retention_ok=len(retention)>0 and all(retention)

    final=all((function_sem_ok,function_struct_ok,guard_ok,retention_ok,persona,metadata_ok))

    print()
    print("Function semantic summary")
    print("-------------------------")
    print(f"Function semantic gain    : {mean_sem:+.3f} => {'PASS' if function_sem_ok else 'FAIL'}")
    print(f"Function structured gain  : {mean_struct:+.3f} => {'PASS' if function_struct_ok else 'FAIL'}")
    print(f"Non-function guard gain   : {guard_mean:+.3f} => {'PASS' if guard_ok else 'FAIL'}")
    print(f"Semantic outcomes         : improved={improved_sem} regressed={regressed_sem}")
    print(f"Structured outcomes       : improved={improved_struct} regressed={regressed_struct}")
    print("Retention                 :","PASS" if retention_ok else "FAIL")
    print("Persona                   :","PASS" if persona else "FAIL")
    print("Metadata                  :","PASS" if metadata_ok else "FAIL")
    print("STATUS                    :","PASS" if final else "FAIL")

    Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report).write_text(json.dumps({
        "version":"v10.12.13",
        "function_results":function_results,
        "mean_function_semantic_gain":mean_sem,
        "mean_function_structured_gain":mean_struct,
        "nonfunction_guard_gain":guard_mean,
        "function_semantic_passed":function_sem_ok,
        "function_structured_passed":function_struct_ok,
        "nonfunction_guard_passed":guard_ok,
        "retention_passed":retention_ok,
        "persona_passed":persona,
        "metadata_passed":metadata_ok,
        "status":"PASS" if final else "FAIL",
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if not final:
        raise SystemExit(2)


if __name__=="__main__":
    main()
