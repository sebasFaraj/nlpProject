"""
T6: turn every literary sentence into a casual one with GPT-5-mini, then add errors.

Usage (from the repo root):
    python src/generate.py --limit 50     # trial run: check cost on the OpenAI usage page
    python src/generate.py                # full run (safe to stop and rerun; it resumes)

Reads:  data/clean/literary_sentences.csv, src/prompts/generation_prompt.txt
Writes: data/synthetic/pairs_raw.csv   (checkpointed every 100 sentences)
        data/synthetic/pairs.csv       (after add_errors.py runs)
        results/generation_log.txt
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

import add_errors
from common import OPENAI_MODEL, call_openai, run_parallel

SENTENCES = Path("data/clean/literary_sentences.csv")
PROMPT = Path("src/prompts/generation_prompt.txt")
RAW_OUT = Path("data/synthetic/pairs_raw.csv")
LOG = Path("results/generation_log.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only do the first N sentences (trial run)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    prompt = PROMPT.read_text(encoding="utf-8").strip()
    df = pd.read_csv(SENTENCES)
    if args.limit:
        df = df.sample(n=args.limit, random_state=42)
    rows = [dict(id=r.id, author=r.author, country=r.country, work=r.work, literary=r.sentence)
            for r in df.itertuples()]

    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    done = run_parallel(rows, lambda r: {"casual_clean": call_openai(prompt, r["literary"])},
                        RAW_OUT, workers=args.workers)

    LOG.parent.mkdir(exist_ok=True)
    LOG.write_text(
        f"model: {OPENAI_MODEL} (reasoning effort: minimal)\n"
        f"date: {datetime.now().isoformat(timespec='minutes')}\n"
        f"prompt file: {PROMPT}\n"
        f"sentences generated: {len(done)} of {len(pd.read_csv(SENTENCES))}\n\n"
        f"prompt:\n{prompt}\n", encoding="utf-8")

    add_errors.main()


if __name__ == "__main__":
    main()
