"""Course-style train_t5.py arguments mapped to the resumable V6 engine.

The one-command project launcher is run_pipeline.py. This wrapper is provided
for users accustomed to the original starter command.
"""
import argparse
import sys
import t5_engine as engine


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--finetune',action='store_true')
    p.add_argument('--experiment_name',default='experiment')
    p.add_argument('--learning_rate',type=float,default=3e-4)
    p.add_argument('--batch_size',type=int,default=8)
    p.add_argument('--test_batch_size',type=int,default=8)
    p.add_argument('--max_n_epochs',type=int,default=10)
    p.add_argument('--patience_epochs',type=int,default=3)
    p.add_argument('--num_beams',type=int,default=4)
    p.add_argument('--num_candidates',type=int,default=4)
    p.add_argument('--max_new_tokens',type=int,default=512)
    p.add_argument('--save_every_batches',type=int,default=200)
    p.add_argument('--dev_select_size',type=int,default=192)
    p.add_argument('--inference_only',action='store_true')
    a=p.parse_args()
    if not a.finetune:
        p.error('V6 currently implements pretrained T5 fine-tuning, not extra credit from scratch; use --finetune')
    if a.experiment_name!='experiment':p.error('The grading output names require experiment_name=experiment')
    b=argparse.Namespace(mode='train',split='dev',batch_size=a.batch_size,eval_batch_size=a.test_batch_size,
        lr=a.learning_rate,epochs=a.max_n_epochs,patience=a.patience_epochs,save_every=a.save_every_batches,
        dev_selection_size=a.dev_select_size,beams=a.num_beams,candidates=a.num_candidates,
        max_source=128,max_target=512,max_new_tokens=a.max_new_tokens)
    engine.validate(b)
    if not a.inference_only:engine.training(b)
    for split in ('dev','test'):
        b.split=split
        engine.generation(b)

if __name__=='__main__':main()