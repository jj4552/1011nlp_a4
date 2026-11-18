"""Run once from the login node before submitting Slurm GPU work."""
import argparse
import ast
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path
from common import ROOT,atomic_json,model_exists,prediction_accuracy


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--skip-data',action='store_true')
    p.add_argument('--skip-smoke',action='store_true',help='Skip real-model CPU one-batch smoke checks')
    a=p.parse_args()
    failures=[]
    os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
    os.environ['XDG_CACHE_HOME']=str(ROOT/'.cache')
    (ROOT/'.cache/huggingface').mkdir(parents=True,exist_ok=True)
    for file in ROOT.rglob('*.py'):
        if 'checkpoints' in file.parts:continue
        try:ast.parse(file.read_text(encoding='utf8'),filename=str(file))
        except SyntaxError as exc:failures.append(str(exc))
    paths={'bert':ROOT/'.venv_bert/bin/python','t5':ROOT/'.venv_t5/bin/python'}
    for name,exe in paths.items():
        if not exe.is_file():failures.append(f'{name}: missing Python environment {exe}');continue
        code='import torch,transformers,numpy; print("OK",torch.__version__,transformers.__version__)'
        if name=='bert':code='import torch,transformers,datasets,numpy; print("OK",torch.__version__,transformers.__version__)'
        command=[str(exe),'-c',code]
        result=subprocess.run(command,capture_output=True,text=True)
        if result.returncode:failures.append(f'{name}: import failed: {result.stderr[-1200:]}')
        else:print(f'{name}: {result.stdout.strip()}',flush=True)
        if result.returncode==0:
            script='part1/main.py' if name=='bert' else 'part2/t5_engine.py'
            result=subprocess.run([str(exe),str(ROOT/script),'--help'],capture_output=True,text=True)
            if result.returncode:failures.append(f'{name}: CLI --help failed: {result.stderr[-1200:]}')
            else:print(f'{name}: CLI accepts arguments',flush=True)
    if not a.skip_data:
        result=subprocess.run([sys.executable,'setup_data.py','--verify-only'],cwd=ROOT,capture_output=True,text=True)
        if result.returncode:failures.append('ATIS data: '+result.stderr[-1200:])
        else:print(result.stdout.strip(),flush=True)
    if not a.skip_smoke and not failures:
        for kind,exe in paths.items():
            result=subprocess.run([str(exe),str(ROOT/'smoke_models.py'),'--model',kind],
                                  capture_output=True,text=True,env=os.environ.copy())
            if result.returncode:
                failures.append(kind+' real CPU smoke: '+result.stderr[-1600:])
            else:
                print(result.stdout.strip(),flush=True)
    q1=ROOT/'part1/out'
    print('Existing Q1 model:',model_exists(q1))
    print('Existing Q1 prediction accuracy:',prediction_accuracy(ROOT/'part1/out_original.txt'))
    atomic_json(ROOT/'PREFLIGHT.json',{'passed':not failures,'errors':failures})
    if failures:
        for x in failures:print('FAIL:',x,flush=True)
        sys.exit(1)
    print('PREFLIGHT PASS (does not verify GPU training accuracy)',flush=True)

if __name__=='__main__':main()