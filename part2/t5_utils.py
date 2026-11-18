"""Starter-compatible lightweight model/optimizer checkpoint API."""
from pathlib import Path
import torch


def initialize_model(args):
    from transformers import T5ForConditionalGeneration,T5Config
    name='google-t5/t5-small'
    model=(T5ForConditionalGeneration.from_pretrained(name) if getattr(args,'finetune',True)
           else T5ForConditionalGeneration(T5Config.from_pretrained(name)))
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')


def initialize_optimizer_and_scheduler(args,model,epoch_length):
    from transformers import get_linear_schedule_with_warmup
    lr=getattr(args,'learning_rate',3e-4)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=getattr(args,'weight_decay',.01))
    n=max(1,epoch_length*getattr(args,'max_n_epochs',10))
    w=min(n-1,epoch_length*getattr(args,'num_warmup_epochs',1))
    scheduler=get_linear_schedule_with_warmup(optimizer,w,n)
    return optimizer,scheduler


def save_model(checkpoint_dir,model,best):
    from common import save_hf_model
    from t5_engine import get_tokenizer
    save_hf_model(Path(checkpoint_dir)/('best' if best else 'last'),model,get_tokenizer())


def load_model_from_checkpoint(args,best):
    from transformers import T5ForConditionalGeneration
    path=Path('checkpoints/t5_best') if best else Path('checkpoints/t5_latest')
    if not path.is_dir():raise FileNotFoundError(path)
    return T5ForConditionalGeneration.from_pretrained(path).to('cuda' if torch.cuda.is_available() else 'cpu')