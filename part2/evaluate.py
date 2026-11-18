import argparse
from utils import compute_metrics

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('-ps','--predicted_sql',required=True)
    p.add_argument('-pr','--predicted_records',required=True)
    p.add_argument('-ds','--development_sql',required=True)
    p.add_argument('-dr','--development_records',required=True)
    a=p.parse_args()
    sql_em,record_em,record_f1,errors=compute_metrics(a.development_sql,a.predicted_sql,a.development_records,a.predicted_records)
    print(f'SQL EM: {sql_em:.6f} | Record EM: {record_em:.6f} | Record F1: {record_f1:.6f} | Errors: {sum(bool(e) for e in errors)}')