"""
T5 step 6: add controlled typing errors to the casual sentences.

With 50% probability per sentence, apply 1-2 of these at random (seed 42):
  - remove accents from 1-3 words (é -> e; ñ stays)
  - make the first letter lowercase
  - drop the opening ¿ or ¡
  - drop the final period

Usage (normally called automatically by generate.py):
    python src/add_errors.py                # data/synthetic/pairs_raw.csv -> data/synthetic/pairs.csv
"""

import random
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

SEED = 42
ERROR_PROB = 0.5
IN_PATH = Path("data/synthetic/pairs_raw.csv")
OUT_PATH = Path("data/synthetic/pairs.csv")

ACCENTED = re.compile(r"[áéíóúÁÉÍÓÚüÜ]")


def strip_accents_word(word):
    # Decompose, drop combining accents, recompose. Protect ñ/Ñ first.
    word = word.replace("ñ", "\0").replace("Ñ", "\1")
    word = "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", word).replace("\0", "ñ").replace("\1", "Ñ")


def remove_accents(s, rng):
    words = s.split(" ")
    idx = [i for i, w in enumerate(words) if ACCENTED.search(w)]
    for i in rng.sample(idx, min(len(idx), rng.randint(1, 3))):
        words[i] = strip_accents_word(words[i])
    return " ".join(words)


def lowercase_first(s, rng):
    for i, c in enumerate(s):
        if c.isalpha():
            return s[:i] + c.lower() + s[i + 1:]
    return s


def drop_opening_mark(s, rng):
    return re.sub(r"^\s*[¿¡]", "", s)


def drop_final_period(s, rng):
    return s[:-1] if s.endswith(".") else s


# (name, function, "can this apply?" check)
ERRORS = [
    ("no_accents", remove_accents, lambda s: bool(ACCENTED.search(s))),
    ("lowercase_start", lowercase_first, lambda s: any(c.isupper() for c in s[:3])),
    ("no_opening_mark", drop_opening_mark, lambda s: s.lstrip()[:1] in ("¿", "¡")),
    ("no_final_period", drop_final_period, lambda s: s.endswith(".")),
]


def add_errors(sentence, rng):
    """Return (sentence_with_errors, list_of_error_names)."""
    if rng.random() >= ERROR_PROB:
        return sentence, []
    possible = [e for e in ERRORS if e[2](sentence)]
    if not possible:
        return sentence, []
    chosen = {e[0] for e in rng.sample(possible, min(len(possible), rng.randint(1, 2)))}
    for name, fn, _check in ERRORS:       # apply in a fixed order
        if name in chosen:
            sentence = fn(sentence, rng)
    return sentence, [e[0] for e in ERRORS if e[0] in chosen]


def main(in_path=IN_PATH, out_path=OUT_PATH):
    df = pd.read_csv(in_path).sort_values("id").reset_index(drop=True)
    rng = random.Random(SEED)
    inputs, kinds = [], []
    for s in df["casual_clean"].fillna(""):
        noisy, names = add_errors(s, rng)
        inputs.append(noisy)
        kinds.append("+".join(names))
    df["casual_input"] = inputs
    df["has_errors"] = [bool(k) for k in kinds]
    df["error_types"] = kinds
    cols = ["id", "author", "country", "work", "literary", "casual_clean", "casual_input", "has_errors", "error_types"]
    df[cols].to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(df)} pairs, {df['has_errors'].mean():.1%} with added errors")


if __name__ == "__main__":
    main(*sys.argv[1:3]) if len(sys.argv) > 1 else main()
