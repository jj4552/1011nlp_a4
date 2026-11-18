"""Offline regression suite: no pretrained weights, GPU, or course data required."""
import argparse
from contextlib import closing
import json
import os
import pickle
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'part1'))
sys.path.insert(0,str(ROOT/'part2'))
from common import atomic_json,atomic_lines,atomic_torch_save,prediction_accuracy
from part1.utils import transform_text,custom_transform
import sql_eval
import part2.t5_engine as t5
import run_pipeline


class Tests(unittest.TestCase):
    def test_ood_is_deterministic_and_preserves_sentiment_words(self):
        example='This wonderful screenplay has fantastic performances. It is not terrible.'
        a=transform_text(example)
        self.assertEqual(a,transform_text(example))
        self.assertEqual(custom_transform({'text':example,'label':1})['label'],1)
        self.assertIn('not',a)
        self.assertIn('terrible',a)
        self.assertNotEqual(example,a)

    def test_atomic_json_and_prediction_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.json'
            atomic_json(p,{'a':2})
            self.assertEqual(json.loads(p.read_text()),{'a':2})
            q=Path(d)/'pred.txt'
            atomic_lines(q,['1','1','0','0'])
            self.assertEqual(prediction_accuracy(q,count=2),1.)
            atomic_lines(q,['1','0','0','0'])
            self.assertEqual(prediction_accuracy(q,count=2),.5)
            atomic_lines(q,['1','0','0'])
            self.assertIsNone(prediction_accuracy(q,count=2))

    def test_sql_metrics_and_pickle_format(self):
        with tempfile.TemporaryDirectory() as d:
            tmp=Path(d)
            db=tmp/'test.db'
            with closing(sqlite3.connect(db)) as conn:
                conn.execute('CREATE TABLE x(v INTEGER)')
                conn.execute('INSERT INTO x VALUES (1),(2)')
                conn.commit()
            with patch.object(sql_eval,'DB_PATH',db):
                qs=['SELECT v FROM x WHERE v=1','SELECT v FROM x WHERE v=2']
                rec,err=sql_eval.save_queries_and_records(qs,tmp/'q.sql',tmp/'q.pkl')
                self.assertEqual(len(rec),2)
                self.assertEqual(err,['',''])
                self.assertEqual(sql_eval.load_queries_and_records(tmp/'q.sql',tmp/'q.pkl')[1],rec)
                self.assertGreater(sql_eval.compute_record_F1(rec,rec),.999999)
                self.assertAlmostEqual(sql_eval.compute_record_F1([[]],[[]]),1.,places=7)
                self.assertLess(sql_eval.compute_record_F1([[('x',)]],[[]]),.0001)
                bad,err=sql_eval.compute_records(['SELECT nope FROM x'])
                self.assertEqual(bad,[[]]);self.assertTrue(err[0])

    def test_stale_bert_outputs_cannot_be_reused(self):
        from common import bert_eval_signature
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'part1';model=p/'out'
            model.mkdir(parents=True)
            (model/'config.json').write_text('{}')
            (model/'pytorch_model.bin').write_bytes(b'first')
            (p/'utils.py').write_text('new transform v6')
            output=p/'out_transformed.txt'
            output.write_text('1\n1\n'*25000)
            meta=p/'out_transformed.txt.meta.json'
            meta.write_text(json.dumps({'signature':'stale'}))
            with patch.object(run_pipeline,'ROOT',root):
                self.assertFalse(run_pipeline.stage_ok('q2_eval'))
                signature=bert_eval_signature(model,'transformed',25000,p/'utils.py')
                meta.write_text(json.dumps({'signature':signature}))
                self.assertTrue(run_pipeline.stage_ok('q2_eval'))
                (model/'pytorch_model.bin').write_bytes(b'second')
                self.assertFalse(run_pipeline.stage_ok('q2_eval'))

    def test_failure_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            tmp=Path(d)
            with patch.object(run_pipeline,'LOGS',tmp/'logs'),patch.object(run_pipeline,'STATUS',tmp/'STATUS.json'),patch.object(run_pipeline,'stage_ok',lambda name:(tmp/name).exists()):
                self.assertFalse(run_pipeline.execute('bad',[sys.executable,'-c','raise RuntimeError("intentional")'],tmp))
                self.assertTrue(run_pipeline.execute('good',[sys.executable,'-c','from pathlib import Path;Path("good").write_text("yes")'],tmp))
                statuses=json.loads((tmp/'STATUS.json').read_text())
                self.assertEqual(statuses['bad']['state'],'FAILED')
                self.assertEqual(statuses['good']['state'],'COMPLETE')
                self.assertIn('RuntimeError', (tmp/'logs/bad.log').read_text())

    def test_torch_checkpoint_state(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'latest.pt'
            model=torch.nn.Linear(2,1)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0001)
            a=model(torch.randn(2,2)).sum();a.backward();optimizer.step()
            atomic_torch_save(p,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':7})
            state=torch.load(p,weights_only=False)
            self.assertEqual(state['step'],7)
            self.assertTrue('state' in state['optimizer'])

    def test_t5_labels_masked_and_shapes(self):
        import torch
        batch=t5.collate([([1,2,3],[4,5]),([1,2],[6])],pad=0)
        self.assertEqual(tuple(batch['input_ids'].shape),(2,3))
        self.assertEqual(tuple(batch['labels'].shape),(2,2))
        self.assertEqual(batch['labels'][1,1].item(),-100)
        self.assertEqual(batch['attention_mask'][1,2].item(),0)

    def test_t5_generation_can_resume_after_failure(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)/'part2'
            (base/'data').mkdir(parents=True)
            (base/'checkpoints/t5_best').mkdir(parents=True)
            (base/'checkpoints/t5_best/config.json').write_text('{}')
            (base/'checkpoints/t5_best/pytorch_model.bin').write_bytes(b'fake-model-weights')
            (base/'data/test.nl').write_text(('hello\n')*432)
            (base/'data/flight_database.db').touch()
            with closing(sqlite3.connect(base/'data/flight_database.db')) as conn:
                conn.execute('CREATE TABLE x(v INTEGER)')
            class Tokenizer:
                pad_token_id=0
                def encode(self,s,**kw):return [2,3,1]
                def batch_decode(self,items,**kw):return ['SELECT 1']*len(items)
            class MockModel:
                def __init__(self,crash_after=None):self.calls=0;self.crash_after=crash_after
                def eval(self):return self
                def generate(self,input_ids,**kw):
                    self.calls+=1
                    if self.crash_after is not None and self.calls>self.crash_after:raise RuntimeError('test preemption')
                    return torch.zeros((len(input_ids)*2,1),dtype=torch.long)
            params=argparse.Namespace(split='test',eval_batch_size=8,beams=2,candidates=2,
                max_new_tokens=24,max_source=20,max_target=32)
            with patch.object(t5,'HERE',base),patch.object(t5,'DB_PATH',base/'data/flight_database.db'),patch.object(sql_eval,'DB_PATH',base/'data/flight_database.db'),patch.object(t5,'model_device',lambda:torch.device('cpu')),patch.object(t5,'get_tokenizer',lambda:Tokenizer()):
                with patch.object(t5,'model_from_best',lambda _:MockModel(crash_after=3)):
                    with self.assertRaisesRegex(RuntimeError,'test preemption'):
                        t5.generation(params)
                progress=base/'results/t5_ft_experiment_test.progress.jsonl'
                self.assertEqual(len(progress.read_text().splitlines()),24)
                resumed=MockModel()
                with patch.object(t5,'model_from_best',lambda _:resumed):
                    t5.generation(params)
                self.assertEqual(len((base/'results/t5_ft_experiment_test.sql').read_text().splitlines()),432)
                with (base/'records/t5_ft_experiment_test.pkl').open('rb') as f:rec,err=pickle.load(f)
                self.assertEqual(len(rec),432)
                self.assertEqual(len(err),432)
                self.assertEqual(resumed.calls,51) # (432-24)/8


if __name__=='__main__':unittest.main(verbosity=2)