"""
T7: drop bad pairs automatically, log why, and pull 100 pairs for the manual check.

Usage:
    python src/filter_pairs.py

Reads:  data/synthetic/pairs.csv
Writes: data/synthetic/pairs_filtered.csv
        results/filter_stats.txt
        results/manual_review_sample.csv   <- open in Google Sheets, split ~33 rows each,
                                              fill in "decision" (keep/reject) and "reason"
                                              (meaning changed / not casual / unnatural / other)
"""

import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

IN_PATH = Path("data/synthetic/pairs.csv")
OUT_PATH = Path("data/synthetic/pairs_filtered.csv")
STATS = Path("results/filter_stats.txt")
REVIEW = Path("results/manual_review_sample.csv")

COMMENTARY = re.compile(r"aquí tienes|aqui tienes|versión casual|version casual|reescrit|"
                        r"here is|here's|^\s*[\"“«].*[\"”»]\s*$", re.IGNORECASE)


def n_words(s):
    return len(str(s).split())


def rules():
    """Each rule returns True if the pair should be DROPPED. Applied in this order."""
    return [
        ("empty", lambda r: not str(r.casual_clean).strip() or str(r.casual_clean) == "nan"),
        ("commentary_or_quotes", lambda r: bool(COMMENTARY.search(str(r.casual_clean)))),
        ("too_similar(>0.85)", lambda r: SequenceMatcher(None, r.literary, r.casual_clean).ratio() > 0.85),
        ("length_ratio(<30%or>150%)",
         lambda r: not (0.30 <= n_words(r.casual_clean) / n_words(r.literary) <= 1.50)),
    ]


def main():
    df = pd.read_csv(IN_PATH)
    df["casual_clean"] = df["casual_clean"].fillna("")
    before = len(df)
    counts = {}
    keep = pd.Series(True, index=df.index)
    for name, rule in rules():
        drop = df[keep].apply(rule, axis=1)
        counts[name] = int(drop.sum())
        keep[drop[drop].index] = False

    out = df[keep].reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)

    lines = [f"Pairs before filtering: {before}"]
    lines += [f"  removed by {k}: {v}" for k, v in counts.items()]
    lines += [f"Pairs after filtering: {len(out)} ({len(out) / before:.1%} kept)",
              "", "Per work after filtering:"]
    lines += [f"  {w}: {n}" for w, n in out["work"].value_counts().items()]
    STATS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    sample = out.sample(n=min(100, len(out)), random_state=42)[["id", "literary", "casual_clean"]]
    sample["reviewer"] = (["A"] * 34 + ["B"] * 33 + ["C"] * 33)[:len(sample)]
    sample["decision"] = ""
    sample["reason"] = ""
    sample.to_csv(REVIEW, index=False)
    print(f"\nManual review sheet: {REVIEW}")
    if len(out) < 3000:
        print("WARNING: fewer than 3,000 pairs left.")


if __name__ == "__main__":
    main()
