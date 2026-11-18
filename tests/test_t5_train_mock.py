"""Mock only Hugging Face model/tokenizer; perform genuine torch gradient/optimizer/scheduler steps.

Verify that an interrupted T5 training run resumes the next minibatch and
finishes producing an actual saved best model + state markers.
"""
import argparse
from contextlib import closing
import json
import os
import pickle
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'part2'))
import sql_eval
import t5_engine as engine


class FakeTokenizer:
    pad_token_id=0
    def encode(self,s,**kwargs):
        return [2,3,1]
    def batch_decode(self,result,**kwargs):
        return ['SELECT 1']*len(result)
    def save_pretrained(self,path):
        Path(path,'tokenizer_config.json').write_text('{}')


class FakeT5(torch.nn.Module):
    calls=0
    crash_at=None
    def __init__(self):
        super().__init__()
        self.embed=torch.nn.Embedding(16,8)
        self.head=torch.nn.Linear(8,1)
    @classmethod
    def from_pretrained(cls,path):
        model=cls()
        weights=Path(path)/'pytorch_model.bin'
        if weights.exists():model.load_state_dict(torch.load(weights,weights_only=False,map_location='cpu'))
        return model
    def save_pretrained(self,path,safe_serialization=False):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        torch.save(self.state_dict(),path/'pytorch_model.bin')
        (path/'config.json').write_text('{}')
    def forward(self,input_ids,attention_mask=None,labels=None):
        FakeT5.calls+=1
        if FakeT5.crash_at is not None and FakeT5.calls==FakeT5.crash_at:
            raise RuntimeError('preempted')
        val=self.head(self.embed(input_ids).mean(1)).mean()
        target=labels.float().masked_fill(labels==-100,0.).sum() / max(1,(labels!=-100).sum().item())
        return types.SimpleNamespace(loss=(val-target).square())
    def generate(self,input_ids,attention_mask=None,num_return_sequences=1,**kw):
        return torch.ones((len(input_ids)*num_return_sequences,2),dtype=torch.long)


def scheduler(opt,num_warmup_steps,num_training_steps):
    return torch.optim.lr_scheduler.LambdaLR(opt,lambda i:1.)


class T5TrainMock(unittest.TestCase):
    def test_checkpoint_resume_and_best_model(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'part2'
            (p/'data').mkdir(parents=True)
            for name,n in [('train',6),('dev',4),('test',3)]:
                (p/'data'/f'{name}.nl').write_text(('find a flight\n')*n)
                if name!='test':(p/'data'/f'{name}.sql').write_text(('SELECT 1\n')*n)
            db=p/'data/flight_database.db'
            with closing(sqlite3.connect(db)) as conn:conn.execute('CREATE TABLE a(x INTEGER)')
            a=argparse.Namespace(mode='train',batch_size=2,eval_batch_size=2,lr=.0003,
                epochs=2,patience=2,save_every=1,dev_selection_size=2,beams=2,candidates=2,
                max_source=32,max_target=32,max_new_tokens=16)
            fake_transformers=types.SimpleNamespace(T5ForConditionalGeneration=FakeT5,
                get_linear_schedule_with_warmup=scheduler)
            FakeT5.calls=0;FakeT5.crash_at=2
            with patch.object(engine,'HERE',p),patch.object(engine,'DB_PATH',db),\
                 patch.object(sql_eval,'DB_PATH',db),patch.object(engine,'model_device',lambda:torch.device('cpu')),\
                 patch.object(engine,'get_tokenizer',lambda:FakeTokenizer()),\
                 patch.dict(sys.modules,{'transformers':fake_transformers}):
                with self.assertRaisesRegex(RuntimeError,'preempted'):
                    engine.training(a)
                checkpoint=p/'checkpoints/t5_last.pt'
                self.assertTrue(checkpoint.is_file())
                state=torch.load(checkpoint,weights_only=False)
                self.assertEqual((state['epoch'],state['batch'],state['steps']),(0,1,1))
                FakeT5.crash_at=None
                engine.training(a)
                self.assertTrue((p/'checkpoints/t5_best/config.json').is_file())
                self.assertTrue((p/'checkpoints/t5_done.json').is_file())
                history=json.loads((p/'checkpoints/t5_history.json').read_text())['epochs']
                self.assertGreaterEqual(len(history),1)
                # One more invocation skips completed model training entirely.
                engine.training(a)


if __name__=='__main__':unittest.main(verbosity=2)