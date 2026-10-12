"""
T5 pilot: run the generation prompt on 20 random sentences so the team can review them.

Usage:  python src/pilot.py
Writes: data/synthetic/pilot_pairs.csv  (fill in "good" = yes/no and a "problem" note)
"""
from pathlib import Path

import pandas as pd

from common import OPENAI_MODEL, call_openai

prompt = Path("src/prompts/generation_prompt.txt").read_text(encoding="utf-8").strip()
df = pd.read_csv("data/clean/literary_sentences.csv").sample(n=20, random_state=7)

rows = []
for i, r in enumerate(df.itertuples(), 1):
    casual = call_openai(prompt, r.sentence)
    print(f"\n[{i}/20] {r.work}\n  LITERARY: {r.sentence}\n  CASUAL:   {casual}")
    rows.append({"id": r.id, "work": r.work, "literary": r.sentence, "casual": casual, "good": "", "problem": ""})

Path("data/synthetic").mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv("data/synthetic/pilot_pairs.csv", index=False)
print(f"\nModel: {OPENAI_MODEL}\nSaved data/synthetic/pilot_pairs.csv - review each row (good: yes/no).")