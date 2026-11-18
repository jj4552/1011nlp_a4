"""Install official ATIS split+DB assets from the public course starter archive."""
import argparse
import io
import shutil
import sqlite3
import urllib.request
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
NEEDED=['part2/data/train.nl','part2/data/train.sql','part2/data/dev.nl','part2/data/dev.sql',
        'part2/data/test.nl','part2/data/flight_database.db','part2/data/flight_database.schema',
        'part2/records/ground_truth_dev.pkl']
URL='https://codeload.github.com/gregdurrett/nyu-dsga1011-a4/zip/refs/heads/main'


def verify():
    for name in NEEDED:
        p=ROOT/name
        if not p.is_file() or p.stat().st_size==0:raise FileNotFoundError(f'Missing official asset: {p}')
    for key,count in [('train',4225),('dev',466),('test',432)]:
        if len((ROOT/f'part2/data/{key}.nl').read_text().splitlines())!=count:raise ValueError(f'Expected {count} ATIS {key} NL')
        if key!='test' and len((ROOT/f'part2/data/{key}.sql').read_text().splitlines())!=count:raise ValueError(f'Expected {count} ATIS {key} SQL')
    conn=sqlite3.connect(str(ROOT/'part2/data/flight_database.db'))
    try:
        n=conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        integrity=conn.execute('PRAGMA quick_check').fetchone()[0]
    finally:conn.close()
    if n!=25 or integrity!='ok':raise ValueError(f'SQLite invalid, tables={n}, check={integrity}')
    print(f'Official ATIS: 4225 train, 466 dev, 432 test; {n} SQLite tables',flush=True)


def install(archive):
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        for rel in NEEDED:
            found=[n for n in names if n.endswith('/'+rel)]
            if len(found)!=1:raise RuntimeError(f'Archive missing/duplicates {rel}')
            dst=ROOT/rel
            if dst.exists() and dst.stat().st_size>0: continue
            dst.parent.mkdir(parents=True,exist_ok=True)
            with z.open(found[0]) as src,dst.open('wb') as out:shutil.copyfileobj(src,out)
            print('Installed',rel,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-zip',type=Path)
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    if a.verify_only:
        verify();return
    if all((ROOT/x).exists() for x in NEEDED):
        verify();return
    if a.source_zip:install(a.source_zip)
    else:
        print('Downloading official course data; best run once on login node',flush=True)
        with urllib.request.urlopen(URL,timeout=120) as f:
            install(io.BytesIO(f.read()))
    verify()

if __name__=='__main__':main()