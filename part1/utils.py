"""Deterministic, label-preserving spelling/keyboard noise for Q2/Q3.

The transformation operates independently for each document, so resumptions,
DataLoader order and Dataset.map caching cannot silently change the text.
"""
import hashlib
import random
import re

NEIGHBORS = {
    'a':'qwsz', 'b':'vghn', 'c':'xdfv', 'd':'serfcx', 'e':'wsdr',
    'f':'drtgvc', 'g':'ftyhbv', 'h':'gyujnb', 'i':'ujko', 'j':'huikmn',
    'k':'jiolm', 'l':'kop', 'm':'njk', 'n':'bhjm', 'o':'iklp', 'p':'ol',
    'q':'wa', 'r':'edft', 's':'awedxz', 't':'rfgy', 'u':'yhji',
    'v':'cfgb', 'w':'qase', 'x':'zsdc', 'y':'tghu', 'z':'asx',
}
TOKEN = re.compile(r"\b[A-Za-z]{4,}\b")
# Keep sentiment-changing / negation words stable. This minimizes label corruption.
PROTECTED = frozenset(('not', 'never', 'none', 'cannot', 'without'))


def transform_text(text, rate=0.46, seed=731):
    key = hashlib.sha256((str(seed) + '\0' + text).encode('utf8')).digest()
    rng = random.Random(int.from_bytes(key[:8], 'big'))
    def replace(m):
        word = m.group()
        if word.lower() in PROTECTED or rng.random() >= rate:
            return word
        positions = [i for i, c in enumerate(word) if c.lower() in NEIGHBORS and 0 < i < len(word)-1]
        if not positions:
            return word
        n = 1 if rng.random() < .88 else 2
        letters = list(word)
        for i in rng.sample(positions, min(n, len(positions))):
            if rng.random() < .14 and len(word) >= 6:
                letters[i] = ''
            else:
                candidate = rng.choice(NEIGHBORS[letters[i].lower()])
                letters[i] = candidate.upper() if letters[i].isupper() else candidate
        return ''.join(letters)
    return TOKEN.sub(replace, text)


def custom_transform(example):
    """The official starter's expected signature."""
    return {**example, 'text': transform_text(example['text'])}