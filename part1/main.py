"""Self-contained BERT training and interruptible Q1/Q2/Q3 evaluation.

CLI:
 python main.py --mode eval --model_dir out --set transformed --output out_transformed.txt
 python main.py --mode q3_train
 python main.py --mode q1_train
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import atomic_json, atomic_lines, atomic_torch_save, model_exists, save_hf_model, rng_state, restore_rng, bert_eval_signature
from utils import custom_transform


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--mode', choices=['q1_train','q3_train','eval'], default=None)
    p.add_argument('--train',action='store_true',help='Original course CLI: fine-tune Q1')
    p.add_argument('--train_augmented',action='store_true',help='Original course CLI: Q3 training')
    p.add_argument('--eval',action='store_true',help='Original course CLI: evaluate clean test')
    p.add_argument('--eval_transformed',action='store_true',help='Original course CLI: evaluate transformed test')
    p.add_argument('--debug_transformation',action='store_true')
    p.add_argument('--set', choices=['clean','transformed'], default='clean')
    p.add_argument('--model_dir', default='out')
    p.add_argument('--output')
    p.add_argument('--batch_size',type=int,default=8)
    p.add_argument('--lr','--learning_rate',type=float,default=None)
    p.add_argument('--epochs','--num_epochs',type=int,default=None)
    p.add_argument('--save_every',type=int,default=300)
    p.add_argument('--debug_train',action='store_true')
    return p.parse_args()


def load_tokenizer():
    from transformers import AutoTokenizer
    if (HERE/'out/tokenizer_config.json').exists():
        return AutoTokenizer.from_pretrained(HERE/'out', use_fast=True)
    return AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)


def raw_data():
    from datasets import load_dataset
    return load_dataset('imdb')


def encode(raw, tokenizer):
    raw=raw.map(lambda x: tokenizer(x['text'], truncation=True, max_length=512), batched=True,
                load_from_cache_file=False, desc='Tokenizing BERT')
    if 'text' in raw.column_names: raw=raw.remove_columns('text')
    if 'label' in raw.column_names: raw=raw.rename_column('label','labels')
    return raw


def training(args):
    import numpy as np
    import torch
    from datasets import concatenate_datasets
    from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, get_linear_schedule_with_warmup
    from torch.utils.data import DataLoader, Subset
    from tqdm.auto import tqdm

    torch.manual_seed(11); np.random.seed(11); random.seed(11)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer=load_tokenizer()
    data=raw_data()
    if args.mode=='q3_train':
        if not model_exists(HERE/'out'):
            raise FileNotFoundError('Q3 requires valid Q1 checkpoint in part1/out/')
        orig=data['train']
        augmented=orig.shuffle(seed=42).select(range(5000)).map(custom_transform, load_from_cache_file=False)
        data_train=concatenate_datasets([orig,augmented])
        initial=HERE/'out'
        out=HERE/'out_augmented'
        lr=args.lr or 5e-6
        epochs=args.epochs or 1
    else:
        data_train=data['train']
        initial='bert-base-cased'
        out=HERE/('out_debug' if args.debug_train else 'out')
        lr=args.lr or 5e-5
        epochs=args.epochs or 3
    if args.debug_train:
        if args.mode=='q3_train': raise ValueError('Q3 must add the full 5000 augmented samples; do not use --debug_train')
        data_train=data_train.shuffle(seed=42).select(range(4000))
    dataset=encode(data_train,tokenizer)
    collate=DataCollatorWithPadding(tokenizer=tokenizer,return_tensors='pt')
    model=AutoModelForSequenceClassification.from_pretrained(initial,num_labels=2).to(device)
    optimizer=torch.optim.AdamW(model.parameters(), lr=lr)
    batch_size=args.batch_size
    n_batches=math.ceil(len(dataset)/batch_size)
    total=n_batches*epochs
    warmup=min(total//10,100) if args.mode=='q3_train' else 0
    scheduler=get_linear_schedule_with_warmup(optimizer,warmup,total)
    path=HERE/'checkpoints'/f'{args.mode}_latest.pt'
    config={'mode':args.mode,'rows':len(dataset),'batch_size':batch_size,'epochs':epochs,'lr':lr,
            'transform':hashlib.sha256((HERE/'utils.py').read_bytes()).hexdigest() if args.mode=='q3_train' else None}
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    epoch0=batch0=global_step=0
    if path.is_file():
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state.get('signature')!=signature:
            raise RuntimeError('Prior checkpoint configuration differs; do not silently restart. '+str(path))
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'])
        epoch0,batch0,global_step=state['epoch'],state['batch'],state['global_step']
        print('Restoring training from:', epoch0,batch0,flush=True)
    def checkpoint(epoch,batch):
        atomic_torch_save(path,{'signature':signature,'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'epoch':epoch,'batch':batch,'global_step':global_step,'rng':rng_state()})
    for epoch in range(epoch0,epochs):
        # Training order is regenerated without consuming dropout RNG.
        generator=torch.Generator().manual_seed(2026+epoch)
        indices=torch.randperm(len(dataset),generator=generator).tolist()
        start=batch0 if epoch==epoch0 else 0
        remain=indices[start*batch_size:]
        loader=DataLoader(Subset(dataset,remain),batch_size=batch_size,shuffle=False,collate_fn=collate)
        model.train()
        rolling=[]
        for batch_i,b in enumerate(tqdm(loader,initial=start,total=n_batches,desc=f'{args.mode} e{epoch+1}'),start=start):
            b={k:v.to(device) for k,v in b.items()}
            optimizer.zero_grad(set_to_none=True)
            loss=model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step(); scheduler.step()
            global_step+=1
            rolling.append(loss.item())
            if (batch_i+1)%args.save_every==0:
                checkpoint(epoch,batch_i+1)
        checkpoint(epoch+1,0)
        print(f'{args.mode} epoch {epoch+1}/{epochs}: loss {sum(rolling)/max(len(rolling),1):.4f}',flush=True)
        batch0=0
    save_hf_model(out,model,tokenizer)
    if args.mode=='q3_train':
        atomic_json(out/'v6_train_config.json',config)
    print('Model saved:',out,flush=True)


def evaluation(args):
    import torch
    from torch.utils.data import DataLoader, Subset
    from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
    from tqdm.auto import tqdm

    if not args.output: raise ValueError('eval requires --output')
    model_dir=(HERE/args.model_dir).resolve()
    if not model_exists(model_dir):
        raise FileNotFoundError(f'Model is missing from {model_dir}')
    tokenizer=load_tokenizer()
    raw=raw_data()['test']
    if args.set=='transformed':
        raw=raw.map(custom_transform,load_from_cache_file=False,desc='Deterministic OOD transform')
    dataset=encode(raw,tokenizer)
    if args.debug_train: dataset=dataset.select(range(1000))
    total=len(dataset)
    dest=(HERE/args.output).resolve()
    dest.parent.mkdir(parents=True,exist_ok=True)
    signature=bert_eval_signature(model_dir,args.set,total,HERE/'utils.py')
    if signature is None: raise FileNotFoundError(f'Missing BERT weights: {model_dir}')
    metadata=dest.with_name(dest.name+'.progress.meta.json')
    progress=dest.with_name(dest.name+'.progress')
    old=json.loads(metadata.read_text()) if metadata.exists() else {}
    if old.get('signature')!=signature:
        progress.unlink(missing_ok=True)
        atomic_json(metadata,{'signature':signature})
    valid=[]
    if progress.is_file():
        for line in progress.read_text().splitlines():
            pair=line.split(',')
            if len(pair)!=2 or pair[0] not in ('0','1') or pair[1] not in ('0','1'):
                raise RuntimeError('Corrupt BERT progress; refusing silent resume')
            valid.append((int(pair[0]),int(pair[1])))
        if len(valid)>total: raise RuntimeError('BERT progress exceeds dataset')
        for i,(_,y) in enumerate(valid):
            if y!=int(dataset[i]['labels']): raise RuntimeError('BERT progress labels misaligned')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    collate=DataCollatorWithPadding(tokenizer=tokenizer,return_tensors='pt')
    pending=DataLoader(Subset(dataset,range(len(valid),total)),batch_size=args.batch_size,collate_fn=collate)
    with progress.open('a',encoding='utf8') as f:
        for b_index,b in enumerate(tqdm(pending,desc=f'BERT {args.set}')):
            labels=b['labels']
            with torch.no_grad():
                pred=model(**{k:v.to(device) for k,v in b.items()}).logits.argmax(-1).cpu()
            for p,y in zip(pred,labels):
                valid.append((int(p),int(y)))
                f.write(f'{int(p)},{int(y)}\n')
            if b_index%10==0:
                f.flush();os.fsync(f.fileno())
        f.flush();os.fsync(f.fileno())
    if len(valid)!=total: raise RuntimeError(f'{len(valid)}/{total} BERT outputs')
    atomic_lines(dest,(str(x) for pair in valid for x in pair))
    score=sum(p==y for p,y in valid)/len(valid)
    atomic_json(dest.with_name(dest.name+'.meta.json'),{'signature':signature,'accuracy':score,'examples':len(valid)})
    print(f'BERT {args.set}: Accuracy={score:.6%}; file={dest}',flush=True)


if __name__=='__main__':
    a=arguments()
    if a.mode is not None:
        if a.mode=='eval':evaluation(a)
        else:training(a)
    else:
        # Preserve official starter invocation forms alongside the V6 --mode CLI.
        if a.debug_transformation:
            samples=raw_data()['test'].shuffle(seed=42).select(range(5))
            for row in samples:
                print('Original:',row['text'][:300])
                print('Transformed:',custom_transform(row)['text'][:300])
            sys.exit(0)
        if not any((a.train,a.train_augmented,a.eval,a.eval_transformed)):
            raise SystemExit('Specify --mode or an original starter flag (--train/--eval/--train_augmented/--eval_transformed)')
        chosen='out_augmented' if a.train_augmented else ('out_debug' if a.train and a.debug_train else Path(a.model_dir).name)
        if a.train:
            a.mode='q1_train';training(a)
            chosen='out_debug' if a.debug_train else 'out'
        if a.train_augmented:
            a.mode='q3_train';training(a)
            chosen='out_augmented'
        if a.eval or a.eval_transformed:
            a.mode='eval';a.model_dir=chosen
            for split in (('clean' if a.eval else None),('transformed' if a.eval_transformed else None)):
                if split is None:continue
                a.set=split
                a.output=f'{chosen}_{"original" if split=="clean" else "transformed"}.txt'
                evaluation(a)
