"""Genuine torch Q3 optimizer/checkpoint recovery with a small fake pretrained model."""
import argparse
import random
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'part1'))
import main as engine

class FakeDataset:
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[i]
    def shuffle(self,seed):
        order=list(range(len(self)));random.Random(seed).shuffle(order)
        return FakeDataset([self.rows[i] for i in order])
    def select(self,indices):return FakeDataset([self.rows[i] for i in indices])
    def map(self,func,**kwargs):return FakeDataset([func(r) for r in self.rows])

class FakeTokenizer:
    def save_pretrained(self,path):Path(path,'tokenizer_config.json').write_text('{}')

class FakeModel(torch.nn.Module):
    calls=0
    crash_at=None
    def __init__(self):
        super().__init__()
        self.head=torch.nn.Linear(2,2)
    @classmethod
    def from_pretrained(cls,path,num_labels=2):
        model=cls();w=Path(path)/'pytorch_model.bin'
        if w.is_file():model.load_state_dict(torch.load(w,weights_only=False))
        return model
    def save_pretrained(self,path,safe_serialization=False):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        torch.save(self.state_dict(),path/'pytorch_model.bin')
        (path/'config.json').write_text('{}')
    def forward(self,input_ids,labels):
        FakeModel.calls+=1
        if FakeModel.crash_at==FakeModel.calls:raise RuntimeError('intentional BERT preemption')
        loss=torch.nn.functional.cross_entropy(self.head(input_ids.float()),labels.long())
        return types.SimpleNamespace(loss=loss)

class FakeCollate:
    def __init__(self,tokenizer,return_tensors):pass
    def __call__(self,batch):
        return {'input_ids':torch.tensor([x['input_ids'] for x in batch]),
                'labels':torch.tensor([x['labels'] for x in batch])}

def prepare_data(ds,tok):
    return FakeDataset([{'input_ids':[1., float(len(x['text'])%3)],'labels':x['label']} for x in ds.rows])

class BERTTrain(unittest.TestCase):
    def test_q3_resume_after_interruption(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'part1';(root/'out').mkdir(parents=True)
            shutil.copyfile(ROOT/'part1/utils.py',root/'utils.py')
            FakeModel().save_pretrained(root/'out')
            FakeTokenizer().save_pretrained(root/'out')
            data=FakeDataset([{'text':'This is a wonderful film with great acting','label':i%2} for i in range(6000)])
            a=argparse.Namespace(mode='q3_train',lr=1e-4,epochs=1,batch_size=1024,save_every=1,debug_train=False)
            def scheduler(optimizer,num_warmup_steps,num_training_steps):
                return torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
            fake_transformers=types.SimpleNamespace(AutoModelForSequenceClassification=FakeModel,
                DataCollatorWithPadding=FakeCollate,get_linear_schedule_with_warmup=scheduler)
            fake_datasets=types.SimpleNamespace(concatenate_datasets=lambda d:FakeDataset(sum([x.rows for x in d],[])))
            FakeModel.calls=0;FakeModel.crash_at=2
            with patch.object(engine,'HERE',root),patch.object(engine,'raw_data',lambda:{'train':data}),\
                 patch.object(engine,'load_tokenizer',lambda:FakeTokenizer()),patch.object(engine,'encode',prepare_data),\
                 patch.dict(sys.modules,{'transformers':fake_transformers,'datasets':fake_datasets}):
                with self.assertRaisesRegex(RuntimeError,'intentional BERT preemption'):
                    engine.training(a)
                last=root/'checkpoints/q3_train_latest.pt'
                state=torch.load(last,weights_only=False)
                self.assertEqual(state['batch'],1)
                self.assertEqual(state['global_step'],1)
                FakeModel.crash_at=None
                engine.training(a)
                self.assertTrue((root/'out_augmented/.complete').is_file())
                self.assertTrue((root/'out_augmented/pytorch_model.bin').is_file())

if __name__=='__main__':unittest.main(verbosity=2)