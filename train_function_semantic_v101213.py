#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_TRY v10.12.13.2 Leakage-Free Function Inference Training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import LanguageModel
from tokenizer_bpe import Tokenizer
from batch_repair_preservation_v10121 import protected_internalized_records
from chat import checkpoint_trained_fingerprints
from semantic_role_generalization_v101210 import training_queries
from function_semantic_decomposition_v101213 import (
    function_training_queries,
    function_inference_prompt,
    function_inference_slot_prompt,
)
from train_semantic_role_generalization_v101210 import (
    RoleQADataset,
    load_role_items,
    masked_loss,
)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--train", default="data/nagato_corpus_semantic_train_v101212.jsonl")
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    p.add_argument("--base-model", default="model/model-gpu-v1.6.2-online.pt")
    p.add_argument("--output", default="model/model-gpu-v1.6.2-online-function-semantic-candidate.pt")
    p.add_argument("--learning-log", default="data/chat_history.jsonl")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--freeze-blocks", type=int, default=2)
    p.add_argument("--function-weight", type=int, default=1)
    p.add_argument("--function-inference-weight", type=int, default=4)
    p.add_argument("--function-generic-weight", type=int, default=3)
    p.add_argument("--nonfunction-replay-weight", type=int, default=2)
    p.add_argument("--preservation-weight", type=int, default=4)
    p.add_argument("--grad-clip", type=float, default=0.5)
    return p.parse_args()


def main():
    args=parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=Tokenizer.load(args.tokenizer)
    model,checkpoint=LanguageModel.load_checkpoint(args.base_model,device=device)

    items=load_role_items(Path(args.train))
    pairs=[]
    function_items=[x for x in items if x.relation=="function"]

    # Base refresh: keep the three established query forms for every proposition.
    for item in items:
        for q in training_queries(item):
            pairs.append((q,item.answer))

    # v10.12.13.1: preservation-balanced replay.
    # Function structure is useful, but v10.12.13 showed that over-weighting it
    # can damage the older generic function representation.  Replay the generic
    # semantic/compact function prompts at comparable or greater weight.
    function_generic_rows=0
    for item in function_items:
        generic_queries=training_queries(item)[1:]
        for _ in range(max(0,args.function_generic_weight)):
            for q in generic_queries:
                pairs.append((q,item.answer))
                function_generic_rows += 1

    function_struct_rows=0
    for item in function_items:
        for _ in range(max(1,args.function_weight)):
            for q in function_training_queries(item):
                pairs.append((q,item.answer))
                function_struct_rows += 1

    # Leakage-free function inference training: the input intentionally hides
    # action/target/purpose.  The model must recover the functional semantics
    # from the subject and learned knowledge rather than receiving gold slots.
    function_inference_rows=0
    for item in function_items:
        inference_queries=(
            function_inference_prompt(item.subject),
            function_inference_slot_prompt(item.subject),
        )
        for _ in range(max(1,args.function_inference_weight)):
            for q in inference_queries:
                pairs.append((q,item.answer))
                function_inference_rows += 1

    # Explicit non-function semantic replay prevents a surgical function update
    # from eroding the semantic space learned by v10.12.12.
    nonfunction_replay_rows=0
    for item in items:
        if item.relation=="function":
            continue
        semantic_queries=training_queries(item)[1:]
        for _ in range(max(0,args.nonfunction_replay_weight)):
            for q in semantic_queries:
                pairs.append((q,item.answer))
                nonfunction_replay_rows += 1

    protected=protected_internalized_records(
        Path(args.learning_log),
        set(checkpoint_trained_fingerprints(checkpoint)),
        set(),
        set(),
    )
    preservation_rows=0
    for record in protected:
        for _ in range(max(0,args.preservation_weight)):
            pairs.append((record.question,record.teacher_answer))
            preservation_rows += 1

    freeze_blocks=max(0,min(args.freeze_blocks,len(model.blocks)))
    for i in range(freeze_blocks):
        for p in model.blocks[i].parameters():
            p.requires_grad=False

    dataset=RoleQADataset(pairs,tokenizer,model.context_length)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=args.learning_rate,weight_decay=0.01)

    print("="*116)
    print(" LLM_TRY v10.12.13.2 Leakage-Free Function Inference Training")
    print("="*116)
    print("Device                :",device)
    if device.type=="cuda":
        print("GPU                   :",torch.cuda.get_device_name(0))
    print("Base model            :",args.base_model)
    print("Total propositions    :",len(items))
    print("Function propositions :",len(function_items))
    print("Function struct weight:",args.function_weight)
    print("Function inference wt. :",args.function_inference_weight)
    print("Function generic wt.  :",args.function_generic_weight)
    print("Non-function replay wt:",args.nonfunction_replay_weight)
    print("Function generic rows :",function_generic_rows)
    print("Function struct rows  :",function_struct_rows)
    print("Function infer rows   :",function_inference_rows)
    print("Nonfunction replay rows:",nonfunction_replay_rows)
    print("Protected concepts    :",len(protected))
    print("Preservation rows     :",preservation_rows)
    print("Total train rows      :",len(pairs))
    print("Learning rate         :",args.learning_rate)
    print("Epoch limit           :",args.epochs)

    best_loss=float("inf")
    best_state=None
    best_epoch=0

    for epoch in range(1,args.epochs+1):
        model.train()
        total=0.0
        batches=0
        for x,y,mask in loader:
            x,y,mask=x.to(device),y.to(device),mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=masked_loss(model,x,y,mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,args.grad_clip)
            optimizer.step()
            total+=float(loss.item()); batches+=1
        mean=total/max(1,batches)
        print(f"epoch={epoch:02d} train={mean:.6f}")
        if mean<best_loss:
            best_loss=mean
            best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No function-semantic candidate produced")

    model.load_state_dict(best_state)
    metadata=checkpoint.get("metadata",{})
    metadata=dict(metadata) if isinstance(metadata,dict) else {}
    metadata.update({
        "function_semantic_version":"v10.12.13.2",
        "function_semantic_parent_version":"v10.12.13.1",
        "function_semantic_parent_version":"v10.12.13",
        "function_semantic_train":args.train,
        "function_semantic_function_count":len(function_items),
        "function_semantic_function_weight":args.function_weight,
        "function_semantic_inference_weight":args.function_inference_weight,
        "function_semantic_inference_rows":function_inference_rows,
        "function_semantic_generic_weight":args.function_generic_weight,
        "function_semantic_nonfunction_replay_weight":args.nonfunction_replay_weight,
        "function_semantic_generic_rows":function_generic_rows,
        "function_semantic_nonfunction_replay_rows":nonfunction_replay_rows,
        "function_semantic_struct_rows":function_struct_rows,
        "function_semantic_preservation_weight":args.preservation_weight,
        "function_semantic_best_epoch":best_epoch,
        "function_semantic_learning_rate":args.learning_rate,
    })
    model.save_checkpoint(
        args.output,optimizer=optimizer,epoch=best_epoch,loss=best_loss,metadata=metadata
    )
    print("Best epoch             :",best_epoch)
    print("Best train loss        :",f"{best_loss:.6f}")
    print("Saved candidate        :",args.output)


if __name__=="__main__":
    main()
