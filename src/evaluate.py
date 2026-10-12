"""
T11: BLEU, ROUGE-L, BERTScore, output length and a Spain-forms check for each system.

Usage:
    python src/evaluate.py                 # all metrics
    python src/evaluate.py --no-bertscore  # skip BERTScore if PyTorch won't install / is too slow

Reads:  results/copy_input.csv, results/baseline_zeroshot.csv, results/baseline_fewshot.csv
Writes: results/metrics.csv, results/spain_forms_hits.csv (check each hit by eye)

ROUGE-L note: we compute it ourselves (longest common subsequence F1, same formula as the
rouge-score package) on lowercased whitespace tokens with punctuation stripped. The rouge-score
package's default tokenizer deletes letters outside a-z, which breaks Spanish (á, ñ), and the
package no longer builds on recent Python versions anyway.
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import sacrebleu

RESULTS = Path("results")
SYSTEMS = {"copy-input": "copy_input.csv", "zero-shot": "baseline_zeroshot.csv", "few-shot": "baseline_fewshot.csv"}
SPAIN_FORMS = re.compile(r"\b(vosotros|vosotras|habéis|tenéis|sois|estáis)\b", re.IGNORECASE)
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def tokenize(s):
    return PUNCT.sub(" ", str(s).lower()).split()


def lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(hyp, ref):
    h, r = tokenize(hyp), tokenize(ref)
    if not h or not r:
        return 0.0
    lcs = lcs_len(h, r)
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(h), lcs / len(r)
    return 2 * p * rec / (p + rec)


def sanity_checks():
    s = "La canción sonaba en la vieja casona de Macondo."
    assert abs(rouge_l(s, s) - 1.0) < 1e-9, "ROUGE-L of a sentence with itself must be 1.0"
    assert rouge_l("canción", "cancion") == 0.0, "'canción' vs 'cancion' must not match"
    assert abs(sacrebleu.corpus_bleu([s], [[s]]).score - 100) < 1e-6
    print("Sanity checks passed (self-score = 1.0, 'canción' != 'cancion').")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bertscore", action="store_true")
    args = ap.parse_args()
    sanity_checks()

    rows, hits = [], []
    for name, fname in SYSTEMS.items():
        path = RESULTS / fname
        if not path.exists():
            print(f"skipping {name}: {path} not found")
            continue
        df = pd.read_csv(path)
        hyps, refs = df["output"].fillna("").astype(str).tolist(), df["literary_reference"].astype(str).tolist()
        row = {
            "system": name,
            "n": len(df),
            "BLEU": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
            "ROUGE-L": round(sum(map(rouge_l, hyps, refs)) / len(df), 4),
            "avg_output_words": round(sum(len(h.split()) for h in hyps) / len(df), 1),
            "avg_reference_words": round(sum(len(r.split()) for r in refs) / len(df), 1),
            "spain_form_outputs": sum(bool(SPAIN_FORMS.search(h)) for h in hyps),
        }
        if not args.no_bertscore:
            from bert_score import score
            _, _, f1 = score(hyps, refs, lang="es", verbose=False)
            row["BERTScore-F1"] = round(f1.mean().item(), 4)
        rows.append(row)
        for i, h in zip(df["id"], hyps):
            if SPAIN_FORMS.search(h):
                hits.append({"system": name, "id": i, "match": SPAIN_FORMS.search(h).group(0), "output": h})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    pd.DataFrame(hits, columns=["system", "id", "match", "output"]).to_csv(RESULTS / "spain_forms_hits.csv", index=False)
    print(metrics.to_string(index=False))
    print(f"\nWrote {RESULTS / 'metrics.csv'}; {len(hits)} Spain-form hits to check by eye in spain_forms_hits.csv")


if __name__ == "__main__":
    main()
