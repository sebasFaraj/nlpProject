"""
T10: run the Gemini baseline (zero-shot and few-shot) on the test set, plus copy-input.

Usage:
    python src/baseline.py --limit 10     # quick check on 10 test sentences (writes *_trial.csv)
    python src/baseline.py                # full test set (safe to stop and rerun; it resumes)

Reads:  data/splits/train.csv (for few-shot examples), data/splits/test.csv
Writes: results/baseline_zeroshot.csv, results/baseline_fewshot.csv, results/copy_input.csv,
        results/baseline_log.txt
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from common import GEMINI_MODEL, call_gemini, run_parallel

TRAIN = Path("data/splits/train.csv")
TEST = Path("data/splits/test.csv")
RESULTS = Path("results")
TEMPERATURE = 0.0
N_SHOTS = 5

INSTRUCTIONS = ("Rewrite the following casual Spanish text as polished literary Spanish, in the style of "
                "South American literary prose. Correct any spelling, accent, or grammar errors. "
                "Keep exactly the same meaning. Return only the rewritten text.")


def few_shot_prompt():
    ex = pd.read_csv(TRAIN).sample(n=N_SHOTS, random_state=42)
    shots = "\n\n".join(f"Casual: {r.casual_input}\nLiterary: {r.literary}" for r in ex.itertuples())
    return f"{INSTRUCTIONS}\n\nExamples:\n\n{shots}", list(ex["id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N test sentences (trial run)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    test = pd.read_csv(TEST)
    suffix = ""
    if args.limit:
        test, suffix = test.head(args.limit), "_trial"
    rows = [dict(id=r.id, casual_input=r.casual_input, literary_reference=r.literary) for r in test.itertuples()]

    # Copy-input: the "do nothing" reference point.
    cp = pd.DataFrame(rows)
    cp["output"] = cp["casual_input"]
    cp.to_csv(RESULTS / f"copy_input{suffix}.csv", index=False)

    fs_prompt, fs_ids = few_shot_prompt()
    systems = {"zeroshot": INSTRUCTIONS, "fewshot": fs_prompt}
    for name, system in systems.items():
        print(f"\n== {name} ==")
        out = run_parallel(rows, lambda r, s=system: {"output": call_gemini(s, r["casual_input"], TEMPERATURE)},
                           RESULTS / f"baseline_{name}{suffix}.csv", workers=args.workers)
        if args.limit:
            for r in out.head(3).itertuples():
                print(f"\n  IN : {r.casual_input}\n  OUT: {r.output}\n  REF: {r.literary_reference}")

    (RESULTS / f"baseline_log{suffix}.txt").write_text(
        f"model: {GEMINI_MODEL}\ndate: {datetime.now().isoformat(timespec='minutes')}\n"
        f"temperature: {TEMPERATURE}\ntest sentences: {len(rows)}\n"
        f"few-shot example ids (from train, seed 42): {fs_ids}\n\n"
        f"ZERO-SHOT PROMPT (system instruction; the casual sentence is the user message):\n{INSTRUCTIONS}\n\n"
        f"FEW-SHOT PROMPT:\n{fs_prompt}\n", encoding="utf-8")
    print("\nDone. Next: python src/evaluate.py")


if __name__ == "__main__":
    main()
