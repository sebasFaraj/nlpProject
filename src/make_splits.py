"""
T8: shuffle with seed 42 and split 80/10/10 into train/val/test.

Usage:
    python src/make_splits.py

Reads:  data/synthetic/pairs_filtered.csv
Writes: data/splits/{train,val,test}.csv, results/split_stats.txt
"""

from pathlib import Path

import pandas as pd

IN_PATH = Path("data/synthetic/pairs_filtered.csv")
OUT_DIR = Path("data/splits")
STATS = Path("results/split_stats.txt")


def main():
    df = pd.read_csv(IN_PATH).sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    splits = {"train": df[:n_train], "val": df[n_train:n_train + n_val], "test": df[n_train + n_val:]}

    # Overlap check: no id may appear in two splits.
    ids = [set(s["id"]) for s in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "id appears in more than one split!"
    assert sum(map(len, ids)) == df["id"].nunique() == n, "duplicate ids in the data"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["Overlap check: passed (no id in more than one split)", ""]
    for name, s in splits.items():
        s.to_csv(OUT_DIR / f"{name}.csv", index=False)
        lines.append(f"{name}: {len(s)} pairs | avg words input {s['casual_input'].str.split().str.len().mean():.1f}"
                     f" | avg words output {s['literary'].str.split().str.len().mean():.1f}"
                     f" | with added errors {s['has_errors'].mean():.0%}")
        lines += [f"    {a}: {k}" for a, k in s["author"].value_counts().items()]
    lines += ["", "Limitation: sentences from the same book appear in every split."]
    STATS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
