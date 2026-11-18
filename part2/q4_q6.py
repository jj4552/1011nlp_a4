"""Q4 statistics and Q6 genuine development-set error analysis."""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))


def lines(path): return Path(path).read_text(encoding='utf8').splitlines()

def percentile(vals,q):
    if not vals:return 0
    s=sorted(vals);p=(len(s)-1)*q;lo=int(p);hi=min(lo+1,len(s)-1)
    return round(s[lo]+(p-lo)*(s[hi]-s[lo]),2)


def q4():
    from transformers import AutoTokenizer
    from t5_engine import MODEL,PREFIX
    from common import atomic_json,atomic_lines
    tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
    def summary(split,processed):
        nl=lines(HERE/'data'/f'{split}.nl')
        sql=lines(HERE/'data'/f'{split}.sql')
        if len(nl)!=len(sql):raise ValueError(f'{split} mismatch')
        def stats(sentences,prefix='',max_len=512):
            lengths=[];vocab=set();trunc=0
            for text in sentences:
                ids=tok.encode(prefix+text,add_special_tokens=True,truncation=False)
                if processed:
                    trunc+=int(len(ids)>max_len)
                    ids=ids[:max_len]
                lengths.append(len(ids));vocab.update(t for t in ids if t not in tok.all_special_ids)
            return {'mean':round(mean(lengths),3),'p50':percentile(lengths,.50),'p95':percentile(lengths,.95),
                    'vocabulary':len(vocab),'truncated':trunc}
        x=stats(nl,PREFIX if processed else '',128)
        y=stats(sql,'',512)
        return {'examples':len(nl),'nl':x,'sql':y}
    result={stage:{split:summary(split,stage=='after_processing') for split in ('train','dev')}
            for stage in ('before_processing','after_processing')}
    result['definitions']={'length':'T5 tokenizer IDs incl special tokens','vocabulary':'distinct non-special T5 token IDs',
                          'after_processing':'source prefix; max source 128, max target 512'}
    atomic_json(HERE/'q4_statistics.json',result)
    report=['# Q4 statistics','', 'T5 tokenizer counts (including EOS in sequence length).','']
    for name in ('before_processing','after_processing'):
        a,b=result[name]['train'],result[name]['dev']
        report+=['## '+name,'','| Statistic | Train | Dev |','|---|---:|---:|']
        for label,key in [('Examples',None),('Mean NL tokens',('nl','mean')),('Mean SQL tokens',('sql','mean')),
                          ('NL vocabulary',('nl','vocabulary')),('SQL vocabulary',('sql','vocabulary')),
                          ('NL P95',('nl','p95')),('SQL P95',('sql','p95'))]:
            v=lambda d:d['examples'] if key is None else d[key[0]][key[1]]
            report.append(f'| {label} | {v(a)} | {v(b)} |')
        report+=['']
    atomic_lines(HERE/'q4_tables.md',report)
    print('Q4: statistics written',flush=True)


def q6():
    from sql_eval import load_queries_and_records
    from common import atomic_json,atomic_lines
    pred=HERE/'results/t5_ft_experiment_dev.sql'
    if not pred.exists():raise FileNotFoundError(pred)
    expected,expected_r,_=load_queries_and_records(HERE/'data/dev.sql',HERE/'records/ground_truth_dev.pkl')
    generated,gen_r,errors=load_queries_and_records(pred,HERE/'records/t5_ft_experiment_dev.pkl')
    if len(expected)!=466 or len(generated)!=466:raise ValueError('Dev must contain 466 examples')
    questions=lines(HERE/'data/dev.nl')
    def portions(q):
        q=re.sub(r'\s+',' ',q).upper().strip()
        out=[]
        for pattern in (r'\bSELECT\s+(.+?)(?:\bFROM\b|$)',
                        r'\bFROM\s+(.+?)(?:\bWHERE\b|\bGROUP\b|\bORDER\b|$)',
                        r'\bWHERE\s+(.+?)(?:\bGROUP\b|\bORDER\b|$)'):
            m=re.search(pattern,q)
            out.append(m.group(1).strip() if m else '')
        return out
    counts=Counter();instances={}
    for i,(a,b) in enumerate(zip(expected,generated)):
        categories=[]
        if errors[i]:
            e=errors[i].lower()
            if 'no such' in e: categories=['unknown column/table']
            elif 'incomplete' in e or 'syntax' in e: categories=['invalid SQL syntax']
            else: categories=['other SQL execution error']
        elif set(expected_r[i])!=set(gen_r[i]):
            x,y=portions(a),portions(b)
            if x[0]!=y[0]:categories.append('projection/aggregation mismatch')
            if x[1]!=y[1]:categories.append('table/JOIN mismatch')
            if x[2]!=y[2]:categories.append('filter mismatch')
            if not categories:categories=['other semantic mismatch']
        for cat in categories:
            counts[cat]+=1
            instances.setdefault(cat,{'index':i,'nl':questions[i],'gold_sql':a,'pred_sql':b,'error':errors[i]})
    result={'total':466,'categories':[{'category':c,'count':n,'example':instances[c]} for c,n in counts.most_common()]}
    atomic_json(HERE/'q6_dev_errors.json',result)
    report=['# Q6 Dev error analysis','', 'Heuristic classes can overlap; review examples manually.','',
            '| Error class | Count / 466 |','|---|---:|']
    for c,n in counts.most_common():report.append(f'| {c} | {n} / 466 |')
    for c,n in counts.most_common():
        e=instances[c];report+=['',f'## {c}', '',f'- Count: {n}/466',f'- Natural language: `{e["nl"]}`',
                               f'- Gold SQL: `{e["gold_sql"]}`',f'- Pred SQL: `{e["pred_sql"]}`']
        if e['error']:report.append(f'- SQLite: `{e["error"]}`')
    atomic_lines(HERE/'q6_error_report.md',report)
    print('Q6: true Dev errors written',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['q4','q6'],required=True)
    a=p.parse_args()
    (q4 if a.mode=='q4' else q6)()