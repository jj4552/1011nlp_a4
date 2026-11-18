"""Compatibility API for the official starter. Main V6 engine uses the same dataset."""
from functools import lru_cache
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from t5_engine import ATISDataset, collate, get_tokenizer

PAD_IDX=0
@lru_cache(None)
def _tok():return get_tokenizer()

class T5Dataset(ATISDataset):
    def __init__(self,data_folder='data',split='train'):
        super().__init__(split,_tok(),128,512)


def normal_collate_fn(batch):
    data=collate(batch,PAD_IDX)
    ids,mask,labels=data['input_ids'],data['attention_mask'],data['labels']
    targets=labels.masked_fill(labels==-100,PAD_IDX)
    dec_inputs=torch.full_like(targets,PAD_IDX)
    dec_inputs[:,1:]=targets[:,:-1]
    return ids,mask,dec_inputs,targets,torch.zeros((len(batch),1),dtype=torch.long)


def test_collate_fn(batch):
    data=collate(batch,PAD_IDX)
    return data['input_ids'],data['attention_mask'],torch.zeros((len(batch),1),dtype=torch.long)


def get_dataloader(batch_size,split):
    return DataLoader(T5Dataset(split=split),batch_size=batch_size,shuffle=(split=='train'),
                      collate_fn=(test_collate_fn if split=='test' else normal_collate_fn))


def load_t5_data(batch_size,test_batch_size):
    return (get_dataloader(batch_size,'train'),get_dataloader(test_batch_size,'dev'),
            get_dataloader(test_batch_size,'test'))


def load_lines(path):return Path(path).read_text().splitlines()


def load_prompting_data(data_folder='data'):
    folder=Path(data_folder)
    return (load_lines(folder/'train.nl'),load_lines(folder/'train.sql'),
            load_lines(folder/'dev.nl'),load_lines(folder/'dev.sql'),load_lines(folder/'test.nl'))