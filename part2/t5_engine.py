"""T5-small ATIS train + dev/test generation, resumable at train minibatches and SQL batches.

This implementation uses the released course evaluator's record-F1 conventions.
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import random
import re
import sqlite3
import sys
from pathlib import Path
from contextlib import closing

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import atomic_json, atomic_lines, atomic_torch_save, model_exists, save_hf_model, rng_state, restore_rng
from sql_eval import DB_PATH, compute_records, compute_record_F1, compute_metrics, save_queries_and_records

MODEL='google-t5/t5-small'
PREFIX='translate English to SQL: '


def cli():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['train','generate'],required=True)
    p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--eval-batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--save-every',type=int,default=200)
    p.add_argument('--dev-selection-size',type=int,default=192)
    p.add_argument('--beams',type=int,default=4)
    p.add_argument('--candidates',type=int,default=4)
    p.add_argument('--max-source',type=int,default=128)
    p.add_argument('--max-target',type=int,default=512)
    p.add_argument('--max-new-tokens',type=int,default=512)
    return p.parse_args()


def validate(a):
    if a.beams<1 or not(1<=a.candidates<=a.beams): raise ValueError('Invalid beams/candidates')
    if min(a.batch_size,a.eval_batch_size,a.epochs,a.patience,a.save_every,a.max_source,a.max_target,a.max_new_tokens)<1: raise ValueError('Positive integer parameters required')
    if a.dev_selection_size < 0: raise ValueError('Negative dev sample')
    if not DB_PATH.is_file(): raise FileNotFoundError(f'Official ATIS DB missing: {DB_PATH}')
    for split,count in [('train',4225),('dev',466),('test',432)]:
        x=HERE/'data'/f'{split}.nl'
        if not x.is_file() or len(x.read_text().splitlines())!=count: raise RuntimeError(f'ATIS {split} NL has wrong count')
        if split!='test':
            y=HERE/'data'/f'{split}.sql'
            if not y.is_file() or len(y.read_text().splitlines())!=count: raise RuntimeError(f'ATIS {split} SQL has wrong count')


def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL,use_fast=True)


def read_lines(p):
    return Path(p).read_text(encoding='utf8').splitlines()


class ATISDataset:
    def __init__(self,split,tokenizer,max_source,max_target):
        self.split=split
        x=read_lines(HERE/'data'/f'{split}.nl')
        y=read_lines(HERE/'data'/f'{split}.sql') if split!='test' else None
        self.inputs=[tokenizer.encode(PREFIX+s,add_special_tokens=True,truncation=True,max_length=max_source) for s in x]
        self.targets=[tokenizer.encode(s,add_special_tokens=True,truncation=True,max_length=max_target) for s in y] if y is not None else None
    def __len__(self): return len(self.inputs)
    def __getitem__(self,i):
        return self.inputs[i],None if self.targets is None else self.targets[i]


def collate(batch,pad=0):
    import torch
    max_x=max(len(a) for a,b in batch)
    ids=torch.full((len(batch),max_x),pad,dtype=torch.long)
    mask=torch.zeros_like(ids)
    labels=None
    if batch[0][1] is not None:
        max_y=max(len(b) for a,b in batch)
        labels=torch.full((len(batch),max_y),-100,dtype=torch.long)
    for i,(x,y) in enumerate(batch):
        ids[i,:len(x)]=torch.tensor(x)
        mask[i,:len(x)]=1
        if labels is not None: labels[i,:len(y)]=torch.tensor(y)
    return {'input_ids':ids,'attention_mask':mask,**({'labels':labels} if labels is not None else {})}


def model_device():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable; train/generate only inside Slurm GPU job')
    return torch.device('cuda')


def model_from_best(device):
    from transformers import T5ForConditionalGeneration
    best=HERE/'checkpoints/t5_best'
    if not model_exists(best): raise FileNotFoundError(f'No completed T5 best model at {best}')
    return T5ForConditionalGeneration.from_pretrained(best).to(device)


def sql_clean(value):
    return str(value).replace('\r',' ').replace('\n',' ').strip()


def syntactically_valid(query):
    if not re.match(r'^\s*(SELECT|WITH)\b',query,re.I): return False
    try:
        # EXPLAIN only validates syntax/schema; it does not execute a potentially expensive query.
        with closing(sqlite3.connect(f'file:{DB_PATH}?mode=ro',uri=True)) as conn:
            conn.execute('EXPLAIN QUERY PLAN '+query).fetchone()
        return True
    except (sqlite3.Error,ValueError):
        return False


def generate_batch(model,tokenizer,batch,device,a):
    import torch
    x=batch['input_ids'].to(device)
    m=batch['attention_mask'].to(device)
    with torch.inference_mode():
        result=model.generate(input_ids=x,attention_mask=m,max_new_tokens=a.max_new_tokens,
                              num_beams=a.beams,num_return_sequences=a.candidates,
                              early_stopping=True)
    texts=tokenizer.batch_decode(result,skip_special_tokens=True)
    answers=[]
    for k in range(0,len(texts),a.candidates):
        qs=[sql_clean(q) for q in texts[k:k+a.candidates]]
        answers.append(next((q for q in qs if syntactically_valid(q)),qs[0]))
    return answers


def generate_subset(model,tokenizer,dataset,indices,device,a):
    from torch.utils.data import DataLoader,Subset
    model.eval()
    loader=DataLoader(Subset(dataset,indices),batch_size=a.eval_batch_size,shuffle=False,collate_fn=lambda b:collate(b,tokenizer.pad_token_id))
    out=[]
    for batch in loader:
        out.extend(generate_batch(model,tokenizer,batch,device,a))
    return out


def dev_ground_truth():
    p=HERE/'records/ground_truth_dev.pkl'
    if p.is_file():
        with p.open('rb') as f: records, errors=pickle.load(f)
        if len(records)==466: return records
    records,errors=compute_records(read_lines(HERE/'data/dev.sql'))
    return records


def training(a):
    import numpy as np
    import torch
    from tqdm.auto import tqdm
    from torch.utils.data import DataLoader,Subset
    from transformers import T5ForConditionalGeneration,get_linear_schedule_with_warmup

    device=model_device()
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    tokenizer=get_tokenizer()
    dataset=ATISDataset('train',tokenizer,a.max_source,a.max_target)
    dev=ATISDataset('dev',tokenizer,a.max_source,a.max_target)
    selection=list(range(min(a.dev_selection_size,len(dev))))
    if selection and len(selection)<len(dev):
        selection=sorted(random.Random(42).sample(list(range(len(dev))),len(selection)))
    gold=dev_ground_truth()
    checkdir=HERE/'checkpoints'
    checkdir.mkdir(exist_ok=True)
    bestdir=checkdir/'t5_best'
    lastfile=checkdir/'t5_last.pt'
    done=checkdir/'t5_done.json'
    config={'model':MODEL,'lr':a.lr,'epochs':a.epochs,'batch':a.batch_size,'patience':a.patience,
            'max_source':a.max_source,'max_target':a.max_target,'beams':a.beams,'candidates':a.candidates,
            'max_new_tokens':a.max_new_tokens,'dev_select':a.dev_selection_size,
            'dataset_hash':hashlib.sha256((HERE/'data/train.sql').read_bytes()).hexdigest()}
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if done.is_file() and json.loads(done.read_text()).get('signature')==signature and model_exists(bestdir):
        print('T5 training already completed; using best checkpoint',flush=True)
        return
    model=T5ForConditionalGeneration.from_pretrained(MODEL).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    n_batches=math.ceil(len(dataset)/a.batch_size)
    total=n_batches*a.epochs
    warmup=min(n_batches//2,max(1,total//10))
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup,num_training_steps=total)
    ep0,bat0,steps=0,0,0
    best=-1.0
    stale=0
    history=[]
    if lastfile.is_file():
        state=torch.load(lastfile,map_location='cpu',weights_only=False)
        if state.get('signature')!=signature:
            raise RuntimeError(f'T5 checkpoint config mismatch; preserve or archive {lastfile} before changing params')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'])
        ep0,bat0,steps=state['epoch'],state['batch'],state['steps']
        best,stale=state['best'],state['stale']
        history=state['history']
        print('T5 recovery:',ep0,bat0,'optimizer steps:',steps,flush=True)
    def checkpoint(ep,batch):
        atomic_torch_save(lastfile,{'signature':signature,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                 'scheduler':scheduler.state_dict(),'rng':rng_state(),'epoch':ep,'batch':batch,
                 'steps':steps,'best':best,'stale':stale,'history':history})
    for epoch in range(ep0,a.epochs):
        gen=torch.Generator().manual_seed(1700+epoch)
        order=torch.randperm(len(dataset),generator=gen).tolist()
        start=bat0 if epoch==ep0 else 0
        remaining=order[start*a.batch_size:]
        loader=DataLoader(Subset(dataset,remaining),batch_size=a.batch_size,shuffle=False,
                          collate_fn=lambda b:collate(b,tokenizer.pad_token_id))
        model.train()
        losses=[]
        for i,b in enumerate(tqdm(loader,initial=start,total=n_batches,desc=f'T5 e{epoch+1}'),start=start):
            b={k:v.to(device) for k,v in b.items()}
            optimizer.zero_grad(set_to_none=True)
            loss=model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step();scheduler.step()
            losses.append(float(loss.detach().item()))
            steps+=1
            if (i+1)%a.save_every==0: checkpoint(epoch,i+1)
        # Persist a full epoch BEFORE slow generation. On preemption eval is repeated,
        # but completed training batches are never lost.
        checkpoint(epoch,n_batches)
        model.eval()
        devloss_sum=0.;ntokens=0
        with torch.inference_mode():
            loader_dev=DataLoader(dev,batch_size=a.eval_batch_size,collate_fn=lambda b:collate(b,tokenizer.pad_token_id))
            for b in loader_dev:
                b={k:v.to(device) for k,v in b.items()}
                n=int((b['labels']!=-100).sum().item())
                devloss_sum+=float(model(**b).loss.item())*n;ntokens+=n
        devloss=devloss_sum/max(ntokens,1)
        subset_queries=generate_subset(model,tokenizer,dev,selection,device,a)
        subset_records,subset_errors=compute_records(subset_queries)
        record_f1=compute_record_F1([gold[i] for i in selection],subset_records)
        print(f'T5 epoch={epoch+1}, train_loss={sum(losses)/max(1,len(losses)):.4f}, dev_loss={devloss:.4f}, dev_subset_RecordF1={record_f1:.5f}',flush=True)
        history.append({'epoch':epoch+1,'dev_loss':devloss,'dev_subset_record_f1':record_f1,'errors':sum(bool(s) for s in subset_errors)})
        if record_f1>best+1e-6:
            # Save selected checkpoint BEFORE changing best value in optimizer state.
            save_hf_model(bestdir,model,tokenizer)
            best=record_f1;stale=0
            print('T5 best updated',best,flush=True)
        else:
            stale+=1
        checkpoint(epoch+1,0)
        atomic_json(checkdir/'t5_history.json',{'signature':signature,'epochs':history,'best_f1_subset':best})
        bat0=0
        if stale>=a.patience:
            print('Early stopping after',epoch+1,flush=True)
            break
    if not model_exists(bestdir): raise RuntimeError('No best T5 checkpoint produced')
    atomic_json(done,{'signature':signature,'best_subset_f1':best,'history':history})
    print('T5 training complete',flush=True)


def generation(a):
    import torch
    from torch.utils.data import DataLoader,Subset
    from tqdm.auto import tqdm
    device=model_device()
    tokenizer=get_tokenizer()
    dataset=ATISDataset(a.split,tokenizer,a.max_source,a.max_target)
    total=len(dataset)
    best=HERE/'checkpoints/t5_best'
    model=model_from_best(device).eval()
    weights=next(iter(best.glob('pytorch_model.bin')),None) or next(iter(best.glob('model.safetensors')),None)
    stamp=weights.stat()
    config={'split':a.split,'size':total,'weight_size':stamp.st_size,'weight_mtime':stamp.st_mtime_ns,
            'beams':a.beams,'candidates':a.candidates,'max_new_tokens':a.max_new_tokens,
            'max_source':a.max_source,'max_target':a.max_target}
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    filebase=f't5_ft_experiment_{a.split}'
    progress=HERE/'results'/f'{filebase}.progress.jsonl'
    meta=progress.with_suffix('.meta.json')
    last_meta=json.loads(meta.read_text()) if meta.exists() else {}
    if last_meta.get('signature')!=signature:
        progress.unlink(missing_ok=True)
        atomic_json(meta,{'signature':signature,'config':config})
    queries=[]
    if progress.is_file():
        for line in progress.read_text(encoding='utf8').splitlines():
            item=json.loads(line)
            if item['i']!=len(queries) or not isinstance(item['sql'],str):
                raise RuntimeError(f'{a.split} generation progress is corrupt')
            queries.append(item['sql'])
        if len(queries)>total: raise RuntimeError('Too many progress records')
    pending=DataLoader(Subset(dataset,range(len(queries),total)),batch_size=a.eval_batch_size,
                       collate_fn=lambda b:collate(b,tokenizer.pad_token_id))
    with progress.open('a',encoding='utf8') as f:
        for batch in tqdm(pending,desc=f'T5 generate {a.split}'):
            predicted=generate_batch(model,tokenizer,batch,device,a)
            for sql in predicted:
                f.write(json.dumps({'i':len(queries),'sql':sql},ensure_ascii=False)+'\n')
                queries.append(sql)
            f.flush();os.fsync(f.fileno())
    if len(queries)!=total: raise RuntimeError(f'T5 {a.split} incomplete {len(queries)}/{total}')
    sql_path=HERE/'results'/f'{filebase}.sql'
    pkl_path=HERE/'records'/f'{filebase}.pkl'
    records,errors=save_queries_and_records(queries,sql_path,pkl_path)
    with pkl_path.open('rb') as f: parsed=pickle.load(f)
    if len(read_lines(sql_path))!=total or len(parsed[0])!=total or len(parsed[1])!=total:
        raise RuntimeError(f'T5 {a.split} outputs corrupted or missing')
    result={'split':a.split,'queries':total,'sql_errors':sum(bool(s) for s in errors)}
    if a.split=='dev':
        gt=HERE/'data/dev.sql'
        ground=HERE/'records/ground_truth_dev.pkl'
        sql_em,record_em,f1,errs=compute_metrics(gt,sql_path,ground if ground.is_file() else None,pkl_path)
        result.update({'sql_em':sql_em,'record_em':record_em,'record_f1':f1})
        print('Dev metrics:',result,flush=True)
    atomic_json(HERE/'results'/f'{filebase}.metrics.json',{'signature':signature,**result})
    print(f'T5 {a.split} complete: {total} SQL / Records',flush=True)


if __name__=='__main__':
    args=cli();validate(args)
    if args.mode=='train': training(args)
    else: generation(args)