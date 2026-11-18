"""One-entry, fail-isolated orchestrator. Run only under an allocated Slurm GPU.

Each stage writes a separate log and structured status immediately. An unrelated
stage can run even after a prior task fails. Re-running the entrypoint is safe.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import ROOT,atomic_json,model_exists,prediction_accuracy,bert_eval_signature,sha256_file

LOGS=ROOT/'logs'
STATUS=ROOT/'STATUS.json'
BERT=ROOT/'.venv_bert/bin/python'
T5=ROOT/'.venv_t5/bin/python'


def read_status():
    try:return json.loads(STATUS.read_text(encoding='utf8'))
    except (OSError,ValueError):return {}


def stage_ok(name):
    p1=ROOT/'part1';p2=ROOT/'part2'
    if name=='q1_model': return model_exists(p1/'out')
    def bert_output_valid(output_name,model_name,split,allow_legacy=False):
        path=p1/output_name
        if prediction_accuracy(path) is None:return False
        meta=path.with_name(path.name+'.meta.json')
        if not meta.exists():
            return allow_legacy and not (p1/model_name/'.complete').exists()
        try:
            found=json.loads(meta.read_text())['signature']
        except (ValueError,KeyError):return False
        expected=bert_eval_signature(p1/model_name,split,25000,p1/'utils.py')
        return found==expected
    if name=='q1_eval':return bert_output_valid('out_original.txt','out','clean',allow_legacy=True)
    if name=='q2_eval':return bert_output_valid('out_transformed.txt','out','transformed')
    if name=='q3_train':
        model=p1/'out_augmented'
        meta=model/'v6_train_config.json'
        if not (model_exists(model) and (model/'.complete').is_file() and meta.is_file()):return False
        try:
            cfg=json.loads(meta.read_text())
            return cfg['transform']==sha256_file(p1/'utils.py') and cfg['rows']==30000 and cfg['epochs']==1 and cfg['lr']==5e-6
        except (KeyError,ValueError):return False
    if name=='q3_clean_eval':return bert_output_valid('out_augmented_original.txt','out_augmented','clean')
    if name=='q3_ood_eval':return bert_output_valid('out_augmented_transformed.txt','out_augmented','transformed')
    if name=='q4':
        p=p2/'q4_statistics.json'
        if not p.is_file() or not (p2/'q4_tables.md').is_file():return False
        try:
            data=json.loads(p.read_text())
            return (data['before_processing']['train']['examples']==4225 and
                    data['before_processing']['dev']['examples']==466)
        except (KeyError,ValueError):return False
    if name=='t5_train':
        best=p2/'checkpoints/t5_best';done=p2/'checkpoints/t5_done.json';train_sql=p2/'data/train.sql'
        if not (model_exists(best) and done.is_file() and train_sql.is_file()):return False
        config={'model':'google-t5/t5-small','lr':3e-4,'epochs':10,'batch':8,'patience':3,
                'max_source':128,'max_target':512,'beams':4,'candidates':4,'max_new_tokens':512,
                'dev_select':192,'dataset_hash':sha256_file(train_sql)}
        sig=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        try:return json.loads(done.read_text())['signature']==sig
        except (KeyError,ValueError):return False
    if name in ('t5_dev','t5_test'):
        split='dev' if name=='t5_dev' else 'test';n=466 if split=='dev' else 432
        sql=p2/f'results/t5_ft_experiment_{split}.sql'
        pkl=p2/f'records/t5_ft_experiment_{split}.pkl'
        metrics=p2/f'results/t5_ft_experiment_{split}.metrics.json'
        if not (sql.is_file() and pkl.is_file() and metrics.is_file()):return False
        if len(sql.read_text().splitlines())!=n:return False
        try:
            import pickle
            with pkl.open('rb') as f:records,errors=pickle.load(f)
            best=p2/'checkpoints/t5_best'
            weights=next((best/x for x in ('pytorch_model.bin','model.safetensors') if (best/x).is_file()),None)
            if weights is None:return False
            st=weights.stat()
            expected={'split':split,'size':n,'weight_size':st.st_size,'weight_mtime':st.st_mtime_ns,
                      'beams':4,'candidates':4,'max_new_tokens':512,'max_source':128,'max_target':512}
            signature=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
            return len(records)==len(errors)==n and json.loads(metrics.read_text()).get('signature')==signature
        except (OSError,ValueError,TypeError,KeyError):return False
    if name=='q6':
        pred=p2/'results/t5_ft_experiment_dev.sql'
        result=p2/'q6_dev_errors.json'
        report=p2/'q6_error_report.md'
        return (pred.is_file() and result.is_file() and report.is_file() and
                min(result.stat().st_mtime_ns,report.stat().st_mtime_ns)>=pred.stat().st_mtime_ns)
    if name=='final':return (ROOT/'RESULTS.md').is_file() and (ROOT/'results_summary.json').is_file()
    raise KeyError(name)


def execute(name,cmd,cwd=None):
    LOGS.mkdir(parents=True,exist_ok=True)
    state=read_status()
    if name!='final' and stage_ok(name):
        print(name+': verified complete, skipping',flush=True)
        state[name]={'state':'COMPLETE','verified_at':time.strftime('%FT%T%z')}
        atomic_json(STATUS,state)
        return True
    state[name]={'state':'RUNNING','started':time.strftime('%FT%T%z'),'command':[str(x) for x in cmd]}
    atomic_json(STATUS,state)
    output=LOGS/(name+'.log')
    print(name+': RUNNING; log='+str(output),flush=True)
    start=time.monotonic()
    try:
        with output.open('a',encoding='utf8') as f:
            f.write('\n=== Starting '+time.strftime('%FT%T%z')+' command: '+repr([str(x) for x in cmd])+' ===\n')
            f.flush()
            proc=subprocess.run([str(x) for x in cmd],cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=False)
        if proc.returncode!=0:
            raise RuntimeError(f'exit code {proc.returncode}; see {output}')
        if not stage_ok(name):
            raise RuntimeError(f'exited 0 but output validation failed; see {output}')
        state[name]={'state':'COMPLETE','seconds':round(time.monotonic()-start,2)}
        print(name+': COMPLETE',flush=True)
        return True
    except BaseException as exc:
        state[name]={'state':'FAILED','seconds':round(time.monotonic()-start,2),'error':repr(exc),'log':str(output)}
        print(name+': FAILED: '+repr(exc),flush=True)
        # Unexpected KeyboardInterrupt/SystemExit should not masquerade as normal.
        if isinstance(exc,(KeyboardInterrupt,SystemExit)):
            atomic_json(STATUS,state)
            raise
        return False
    finally:
        atomic_json(STATUS,state)


def finalize():
    from common import atomic_json,atomic_lines
    p1=ROOT/'part1';p2=ROOT/'part2'
    results={}
    for name,path in [('q1_clean',p1/'out_original.txt'),('q2_transformed',p1/'out_transformed.txt'),
                      ('q3_clean',p1/'out_augmented_original.txt'),('q3_transformed',p1/'out_augmented_transformed.txt')]:
        results[name]=prediction_accuracy(path)
    if results['q1_clean'] is not None and results['q2_transformed'] is not None:
        results['q2_drop_pp']=100*(results['q1_clean']-results['q2_transformed'])
    metric=p2/'results/t5_ft_experiment_dev.metrics.json'
    results['t5_dev']=json.loads(metric.read_text()) if metric.is_file() else None
    metric=p2/'results/t5_ft_experiment_test.metrics.json'
    results['t5_test']=json.loads(metric.read_text()) if metric.is_file() else None
    results['stage_status']=read_status()
    atomic_json(ROOT/'results_summary.json',results)
    report=['# Actual A4 run results','', 'No hidden Test Record F1 can be claimed without hidden labels.','',
            '| Metric | Observed |','|---|---:|']
    for name in ('q1_clean','q2_transformed','q2_drop_pp','q3_clean','q3_transformed'):
        value=results.get(name)
        report.append(f'| {name} | {value:.4f}'+(' pp' if name=='q2_drop_pp' else '')+' |' if value is not None else f'| {name} | INCOMPLETE |')
    if results['t5_dev']:
        for metric in ('record_f1','record_em','sql_em'):
            if metric in results['t5_dev']:report.append(f'| T5 Dev {metric} | {results["t5_dev"][metric]:.4f} |')
    report+=['','## Stage status','']
    for key,value in read_status().items():
        if key!='final':report.append(f'- {key}: {value.get("state","UNKNOWN")}')
    atomic_lines(ROOT/'RESULTS.md',report)
    print('Wrote RESULTS.md and results_summary.json',flush=True)


def run():
    p=argparse.ArgumentParser()
    p.add_argument('--only',nargs='*',default=None,help='Optional stages, for debugging')
    a=p.parse_args()
    if not BERT.is_file():print(f'Warning: missing BERT venv {BERT}',flush=True)
    if not T5.is_file():print(f'Warning: missing T5 venv {T5}',flush=True)
    from common import atomic_json
    # Do T5 first: previous attempts wasted a GPU without ever reaching T5.
    def cmd(name,command,cwd):
        if a.only is None or name in a.only:return execute(name,command,cwd)
        return stage_ok(name)
    p1=ROOT/'part1';p2=ROOT/'part2'
    cmd('t5_train',[T5,'t5_engine.py','--mode','train'],p2)
    if stage_ok('t5_train'):
        cmd('t5_dev',[T5,'t5_engine.py','--mode','generate','--split','dev'],p2)
        cmd('t5_test',[T5,'t5_engine.py','--mode','generate','--split','test'],p2)
    else:print('T5 predictions blocked: no completed best T5 model',flush=True)
    if not stage_ok('q1_model'):
        cmd('q1_model',[BERT,'main.py','--mode','q1_train'],p1)
    if stage_ok('q1_model'):
        cmd('q1_eval',[BERT,'main.py','--mode','eval','--model_dir','out','--set','clean','--output','out_original.txt'],p1)
        cmd('q2_eval',[BERT,'main.py','--mode','eval','--model_dir','out','--set','transformed','--output','out_transformed.txt'],p1)
        cmd('q3_train',[BERT,'main.py','--mode','q3_train'],p1)
    else:print('Q1 model missing: Q1-Q3 evaluations/training blocked',flush=True)
    if stage_ok('q3_train'):
        cmd('q3_clean_eval',[BERT,'main.py','--mode','eval','--model_dir','out_augmented','--set','clean','--output','out_augmented_original.txt'],p1)
        cmd('q3_ood_eval',[BERT,'main.py','--mode','eval','--model_dir','out_augmented','--set','transformed','--output','out_augmented_transformed.txt'],p1)
    cmd('q4',[T5,'q4_q6.py','--mode','q4'],p2)
    if stage_ok('t5_dev'):
        cmd('q6',[T5,'q4_q6.py','--mode','q6'],p2)
    else:print('Q6 blocked pending valid Dev predictions',flush=True)
    cmd('final',[sys.executable,'run_pipeline.py','--only-finalize'],ROOT)
    print('Run summary:',json.dumps({k:v.get('state') for k,v in read_status().items()},sort_keys=True),flush=True)
    if any(v.get('state')=='FAILED' for k,v in read_status().items() if k!='final'):
        sys.exit(1)


if __name__=='__main__':
    if '--only-finalize' in sys.argv:
        finalize()
    else:
        run()