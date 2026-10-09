"""
Written by Sebastian, with help from claude for the regexes

T4: clean ONE book and add a sample of its good sentences to the dataset.

Usage (from the repo root, once per book):
    python src/extract_sentences.py data/raw/borges_ficciones.txt \
        --author "Jorge Luis Borges" --country Argentina --work "Ficciones"

Optional:
    --max 600     how many sentences to sample from this book (default 600)

Writes:
    data/clean/literary_sentences.csv   all books' sampled sentences (this book's rows are added or replaced)
    data/clean/filter_counts.json       how many sentences each filter removed, per book
    results/data_stats.txt              summary stats for the whole dataset (rebuilt every run)

Running the same book twice replaces its rows instead of duplicating them,
so it is safe to tweak a filter and rerun.
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy

CSV_PATH = Path("data/clean/literary_sentences.csv")
FILTER_PATH = Path("data/clean/filter_counts.json")
STATS_PATH = Path("results/data_stats.txt")

MIN_WORDS, MAX_WORDS = 8, 40
SEED = 42

STARTS_OK = re.compile(r"^[A-ZÁÉÍÓÚÜÑ¿¡]")
ENDS_OK = re.compile(r"[.!?…]$")
DIALOGUE = re.compile(r"[—–]|^[«\"“'‘]")    # dialogue dashes, or opening with a quote
JUNK_CHARS = re.compile(r"[\d\[\]{}<>*_|#@/\\=+]")  # digits, footnote marks, PDF artifacts


# Cleaning Raw Text

def is_heading(line):
    """A short line that is mostly capitals ("CAPÍTULO I", "PRIMERA PARTE") is a heading."""
    letters = [c for c in line if c.isalpha()]
    return 0 < len(line.strip()) <= 60 and bool(letters) and \
        sum(c.isupper() for c in letters) / len(letters) > 0.8

def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("­", "")       # soft hyphens
    text = "\n".join("" if is_heading(l) else l for l in text.split("\n"))  # drop chapter headings
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)                   # re-join words split at line end
    text = re.sub(r"\n(?=\s*[—–])", "\n\n", text)                  # a dialogue line starts a new paragraph
    paragraphs = re.split(r"\n\s*\n", text)                        # blank line = paragraph break
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in paragraphs]  # line wraps -> spaces
    return [p for p in paragraphs if p]


def chunk(paragraphs, size=50_000):
    """spaCy has a length limit, so this cut very long paragraphs at sentence-like boundaries."""
    for p in paragraphs:
        if len(p) <= size:
            yield p
            continue
        piece = ""
        for part in re.split(r"(?<=[.!?…])\s+", p):
            if len(piece) + len(part) > size and piece:
                yield piece
                piece = ""
            piece += part + " "
        if piece.strip():
            yield piece.strip()


# Splitting into sentences and filtering

def split_sentences(paragraphs):
    # Only the parser is needed for sentence boundaries; skip the rest for speed.
    nlp = spacy.load("es_core_news_sm", exclude=["ner", "lemmatizer", "attribute_ruler"])
    for doc in nlp.pipe(chunk(paragraphs), batch_size=32):
        for sent in doc.sents:
            yield sent.text.strip()


def reject_reason(s):
    """Return why a sentence is rejected, or None if it's kept. Checked in this order."""
    if DIALOGUE.search(s):
        return "dialogue"
    if JUNK_CHARS.search(s):
        return "digits_or_symbols"
    n = len(s.split())
    if n < MIN_WORDS:
        return "too_short"
    if n > MAX_WORDS:
        return "too_long"
    if not STARTS_OK.match(s) or not ENDS_OK.search(s):
        return "incomplete"
    letters = [c for c in s if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.5:
        return "all_caps"
    return None


# Saving results and statistics

def write_stats(df, filter_counts):
    lines = ["LITERARY SENTENCE DATASET", "=" * 60,
             f"Total sentences: {len(df)}",
             f"Works: {df['work'].nunique()}   Authors: {df['author'].nunique()}   "
             f"Countries: {df['country'].nunique()}",
             f"Words per sentence: mean {df['n_words'].mean():.1f}, "
             f"median {df['n_words'].median():.0f}, min {df['n_words'].min()}, max {df['n_words'].max()}",
             "", "PER WORK", "-" * 60]
    
    for (author, work), g in df.groupby(["author", "work"]):
        fc = filter_counts.get(work, {})
        lines.append(f"{author} - {work}")
        lines.append(f"  raw sentences: {fc.get('raw', '?')}   passed filters: {fc.get('kept', '?')}   "
                     f"sampled: {len(g)}   mean words: {g['n_words'].mean():.1f}")
        removed = {k: v for k, v in fc.items() if k not in ("raw", "kept", "duplicate_free")}
        if removed:
            lines.append("  removed: " + ", ".join(f"{k} {v}" for k, v in removed.items()))
    lines += ["", "PER COUNTRY", "-" * 60]

    for country, n in df["country"].value_counts().items():
        lines.append(f"{country}: {n} ({n / len(df):.0%})")
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path, help="data/raw/<author>_<work>.txt")
    ap.add_argument("--author", required=True)
    ap.add_argument("--country", required=True)
    ap.add_argument("--work", help="title of the work (defaults to the file name)")
    ap.add_argument("--max", type=int, default=600, help="sentences to sample from this work")
    args = ap.parse_args()
    work = args.work or args.file.stem

    # Clean, split, filter
    raw = list(split_sentences(clean_text(args.file.read_text(encoding="utf-8"))))
    counts = Counter()
    good = []
    for s in raw:
        reason = reject_reason(s)
        counts[reason or "kept_before_dedup"] += 1
        if reason is None:
            good.append(s)
    unique = list(dict.fromkeys(good))           # remove exact duplicates, keep order
    counts["duplicate"] = len(good) - len(unique)
    del counts["kept_before_dedup"]

    # Sample (a random spread across the whole book, reproducible with the seed)
    pool = pd.DataFrame({"sentence": unique})
    sample = pool.sample(n=min(args.max, len(pool)), random_state=SEED).sort_index()
    sample = sample.assign(author=args.author, country=args.country, work=work,
                           n_words=sample["sentence"].str.split().str.len())
    sample.insert(0, "id", [f"{args.file.stem}_{i:04d}" for i in range(len(sample))])

    # 3. Add to the dataset, replacing any earlier rows for this work
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH) if CSV_PATH.exists() else pd.DataFrame()
    if not df.empty:
        df = df[df["work"] != work]
    cols = ["id", "author", "country", "work", "sentence", "n_words"]
    df = pd.concat([df, sample[cols]], ignore_index=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    filter_counts = json.loads(FILTER_PATH.read_text(encoding="utf-8")) if FILTER_PATH.exists() else {}
    filter_counts[work] = {"raw": len(raw), "kept": len(unique), **dict(counts)}
    FILTER_PATH.write_text(json.dumps(filter_counts, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stats(df, filter_counts)

    # Report
    print(f"{work}: {len(raw)} raw sentences -> {len(unique)} passed filters -> {len(sample)} sampled")
    print("  removed: " + ", ".join(f"{k} {v}" for k, v in counts.most_common() if v))
    if len(unique) < args.max:
        print(f"  note: only {len(unique)} good sentences, fewer than --max {args.max}")
    print(f"Dataset now has {len(df)} sentences from {df['work'].nunique()} work(s).")
    print("\n5 random examples from this work:")
    for s in sample["sentence"].sample(n=min(5, len(sample)), random_state=1):
        print("  -", s)


if __name__ == "__main__":
    main()