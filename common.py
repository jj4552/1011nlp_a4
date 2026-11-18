"""Small, dependency-free IO and checkpoint helpers."""
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_lines(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with tmp.open('w', encoding='utf8') as f:
            for v in values:
                f.write(str(v).replace('\n', ' ').replace('\r', ' ') + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_torch_save(path, state):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_hf_model(directory, model, tokenizer):
    """Publish a complete model directory only after save_pretrained has finished."""
    import shutil
    target = Path(directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f'.writing-{os.getpid()}')
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        model.save_pretrained(tmp, safe_serialization=False)
        tokenizer.save_pretrained(tmp)
        (tmp / '.complete').write_text('ok\n')
        # An old working model is never deleted until a new model is finished.
        backup = target.with_name(target.name + '.previous')
        if target.exists():
            shutil.rmtree(backup, ignore_errors=True)
            os.replace(target, backup)
        os.replace(tmp, target)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def model_exists(path):
    p = Path(path)
    return (p / 'config.json').is_file() and any((p / f).is_file() for f in ('pytorch_model.bin', 'model.safetensors'))


def prediction_accuracy(path, count=25000):
    p = Path(path)
    if not p.is_file():
        return None
    with p.open(encoding='utf8') as f:
        tokens = [s.strip() for s in f]
    if len(tokens) != 2 * count or any(t not in {'0', '1'} for t in tokens):
        return None
    return sum(tokens[i] == tokens[i + 1] for i in range(0, len(tokens), 2)) / count


def rng_state():
    import numpy as np
    import torch
    result = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        result['cuda'] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def bert_eval_signature(model_dir, split, count, transform_code):
    """Shared evaluator/runner signature, so stale predictions cannot be reused."""
    p = Path(model_dir).resolve()
    w = next((p / x for x in ('pytorch_model.bin','model.safetensors') if (p / x).is_file()),None)
    if w is None: return None
    st = w.stat()
    data = {'model':str(p),'weight_size':st.st_size,'weight_mtime':st.st_mtime_ns,
            'set':split,'count':count,
            'transform_hash':sha256_file(transform_code)}
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()