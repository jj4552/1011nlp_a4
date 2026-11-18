"""One-example CPU smoke test. Run on login node BEFORE requesting GPU.

Uses real cached/pretrained model weights; requires no data downloads beyond model cache.
"""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'part2'))


def bert_smoke():
    import torch
    from datasets import load_dataset
    from part1.utils import custom_transform
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    p=ROOT/'part1/out'
    if not (p/'config.json').is_file():
        print('BERT Q1 model not found; actual Q1 training needed')
        return
    token_path=p if (p/'tokenizer_config.json').exists() else 'bert-base-cased'
    tokenizer=AutoTokenizer.from_pretrained(token_path,use_fast=True)
    model=AutoModelForSequenceClassification.from_pretrained(p)
    model.eval()
    # Exercise the same real IMDB source + custom transformation as Q2.
    sample=load_dataset('imdb',split='test[:1]')
    transformed=sample.map(custom_transform,load_from_cache_file=False)
    assert int(transformed[0]['label'])==int(sample[0]['label'])
    with torch.inference_mode():
        out=model(**tokenizer([transformed[0]['text']],return_tensors='pt',truncation=True,max_length=512))
    assert out.logits.shape==(1,2)
    print('PASS: real BERT weights load and one-example CPU forward',flush=True)


def t5_smoke():
    import torch
    from transformers import AutoTokenizer,T5ForConditionalGeneration
    from part2.t5_engine import ATISDataset,collate,MODEL
    from torch.utils.data import DataLoader
    tokenizer=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
    model=T5ForConditionalGeneration.from_pretrained(MODEL)
    model.eval()
    dataset=ATISDataset('train',tokenizer,128,512)
    assert len(dataset)==4225
    batch=collate([dataset[0]],tokenizer.pad_token_id)
    with torch.inference_mode():
        loss=model(**batch).loss
        generated=model.generate(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],max_new_tokens=4)
    assert torch.isfinite(loss)
    assert generated.shape[0]==1
    print('PASS: real T5 pretrained weights load, one-sample forward and generation',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['bert','t5'],required=True)
    a=p.parse_args()
    if a.model=='bert':bert_smoke()
    else:t5_smoke()