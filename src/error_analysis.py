"""
T12: error analysis on 30 few-shot outputs.

Step 1  python src/error_analysis.py sample
        -> results/error_analysis.csv with 30 random few-shot outputs and an empty "category" column.
        Open it (Google Sheets is fine), and give each row ONE category:
            good | register_collapse | regional_flattening | meaning_drift | over_embellishment | missed_error
        Mark your 3 strongest and 3 weakest examples with "strong" / "weak" in the "pick" column.
        Save it back as CSV in the same place.

Step 2  python src/error_analysis.py count
        -> results/error_categories.csv, results/error_categories.png (bar chart for slide 9),
           results/examples_for_slides.txt (the 6 picked examples, input/output/reference together)
"""

import sys
from pathlib import Path

import pandas as pd

RESULTS = Path("results")
SHEET = RESULTS / "error_analysis.csv"
CATEGORIES = ["good", "register_collapse", "regional_flattening", "meaning_drift", "over_embellishment", "missed_error"]
LABELS = ["Good", "Register\ncollapse", "Regional\nflattening", "Meaning\ndrift", "Over-\nembellishment", "Missed\nerror"]


def sample():
    df = pd.read_csv(RESULTS / "baseline_fewshot.csv").sample(n=30, random_state=42)
    df["category"] = ""
    df["pick"] = ""
    df["note"] = ""
    df.to_csv(SHEET, index=False)
    print(f"Wrote {SHEET}. Categories: {', '.join(CATEGORIES)}")


def count():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(SHEET).fillna("")
    df["category"] = df["category"].str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    bad = df[~df["category"].isin(CATEGORIES)]
    if len(bad):
        print(f"WARNING: {len(bad)} rows have no/unknown category: {bad['category'].unique().tolist()}")
    counts = df["category"].value_counts().reindex(CATEGORIES, fill_value=0)
    counts.to_csv(RESULTS / "error_categories.csv", header=["count"])
    print(counts.to_string())

    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    colors = ["#2a9d8f"] + ["#c8553d"] * 5
    bars = ax.bar(LABELS, counts.values, color=colors)
    ax.bar_label(bars, padding=2)
    ax.set_ylabel(f"outputs (of {len(df)})")
    ax.set_title("Few-shot Gemini baseline: main error per output")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "error_categories.png")

    lines = []
    for kind in ("strong", "weak"):
        for r in df[df["pick"].str.lower().str.strip() == kind].itertuples():
            lines += [f"[{kind.upper()}] {r.id}  ({r.category})",
                      f"  input:     {r.casual_input}", f"  baseline:  {r.output}",
                      f"  reference: {r.literary_reference}", f"  note:      {r.note}", ""]
    (RESULTS / "examples_for_slides.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote error_categories.png and examples_for_slides.txt")


if __name__ == "__main__":
    {"sample": sample, "count": count}[sys.argv[1] if len(sys.argv) > 1 else "sample"]()
