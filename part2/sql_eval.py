"""ATIS SQLite scoring compatible with the released course evaluator."""
import os
import hashlib
import json
import pickle
import sqlite3
import time
from contextlib import closing
from pathlib import Path
import numpy as np

DB_PATH = Path(__file__).resolve().parent / 'data' / 'flight_database.db'


def read_queries(path):
    return Path(path).read_text(encoding='utf8').splitlines()


def compute_records(queries):
    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)
    records, errors = [], []
    with closing(sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True, timeout=5)) as conn:
        for q in queries:
            started = time.monotonic()
            # Bound individual malformed/expensive generated queries. The SQLite
            # progress handler interrupts a query rather than hanging a GPU job.
            conn.set_progress_handler(lambda: int(time.monotonic()-started > 15.), 10000)
            try:
                if not q.lstrip().upper().startswith(('SELECT ', 'WITH ')):
                    raise ValueError('Only SELECT/WITH allowed')
                rows = conn.execute(q).fetchall()
                records.append(rows)
                errors.append('')
            except Exception as exc:
                records.append([])
                errors.append(f'{type(exc).__name__}: {exc}')
            finally:
                conn.set_progress_handler(None, 0)
    return records, errors


def save_queries_and_records(queries, sql_path, record_path):
    """Publish course-format files, resuming SQLite execution in 16-query chunks.

    The private progress file includes a digest of the query list so stale
    records from an earlier checkpoint can never be reused for changed SQL.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from common import atomic_lines
    sql_path = Path(sql_path)
    record_path = Path(record_path)
    sql_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(list(queries),ensure_ascii=False).encode('utf8')).hexdigest()
    progress = record_path.with_name(record_path.name + '.records.progress.pkl')
    records, errors = [], []
    if progress.exists():
        with progress.open('rb') as f:
            checkpoint = pickle.load(f)
        if checkpoint.get('digest') == digest:
            records = checkpoint['records']
            errors = checkpoint['errors']
            if len(records) != len(errors) or len(records)>len(queries):
                raise RuntimeError('Corrupt SQL records progress; refusing silent resume')
        else:
            progress.unlink()
    for start in range(len(records),len(queries),16):
        result, failures = compute_records(queries[start:start+16])
        records.extend(result);errors.extend(failures)
        tmp = progress.with_name(progress.name+'.writing')
        try:
            with tmp.open('wb') as f:
                pickle.dump({'digest':digest,'records':records,'errors':errors},f,protocol=4)
                f.flush();os.fsync(f.fileno())
            os.replace(tmp,progress)
        finally:
            tmp.unlink(missing_ok=True)
    if len(records)!=len(queries) or len(errors)!=len(queries):
        raise RuntimeError('Incomplete SQL record computation')
    sql_tmp = sql_path.with_name(sql_path.name + '.writing')
    record_tmp = record_path.with_name(record_path.name + '.writing')
    atomic_lines(sql_tmp, queries)
    with record_tmp.open('wb') as f:
        pickle.dump((records, errors), f, protocol=4)
        f.flush();os.fsync(f.fileno())
    os.replace(record_tmp, record_path)
    os.replace(sql_tmp, sql_path)
    progress.unlink(missing_ok=True)
    return records, errors


def load_queries_and_records(sql_path, records_path=None):
    queries = read_queries(sql_path)
    if records_path is None:
        records, errors = compute_records(queries)
    else:
        with open(records_path, 'rb') as f:
            records, errors = pickle.load(f)
    if len(queries) != len(records) or len(queries) != len(errors):
        raise ValueError('SQL/record/error counts do not match')
    return queries, records, errors


def compute_sql_exact_match(gt, pred):
    if len(gt) != len(pred): raise ValueError('SQL lengths differ')
    return float(sum(a == b for a,b in zip(gt,pred)) / max(1,len(gt)))


def compute_record_exact_match(gt, pred):
    if len(gt) != len(pred): raise ValueError('Record lengths differ')
    return float(sum(set(a) == set(b) for a,b in zip(gt,pred)) / max(1,len(gt)))


def compute_record_F1(gt, pred):
    if len(gt) != len(pred): raise ValueError('Record lengths differ')
    scores = []
    for gold, generated in zip(gt, pred):
        g, p = set(gold), set(generated)
        precision = len(g & p) / len(p) if p else 1.
        recall = len(g & p) / len(g) if g else 1.
        scores.append(2*precision*recall/(precision+recall+1e-8))
    return float(np.mean(scores)) if scores else 0.


def compute_metrics(gt_path, model_path, gt_query_records=None, model_query_records=None):
    gt, gt_records, _ = load_queries_and_records(gt_path, gt_query_records)
    pred, pred_records, errors = load_queries_and_records(model_path, model_query_records)
    return (compute_sql_exact_match(gt,pred), compute_record_exact_match(gt_records,pred_records),
            compute_record_F1(gt_records,pred_records), errors)