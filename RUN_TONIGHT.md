# Run tonight: T6 → T12

T1–T4 are done (4,200 sentences in `data/clean/literary_sentences.csv`). Everything below is
scripted. One person with the `.env` keys runs the commands, from the repo root, in order.

**Before starting:** if your friend's T5 prompt differs from `src/prompts/generation_prompt.txt`,
paste their final version into that file. Then:

```
pip install -r requirements.txt
python src/test_api.py
```

## 1. Generate casual versions (T6), ~20–40 min

```
python src/generate.py --limit 50
```
Check the OpenAI usage page for the cost of 50, multiply by 84. Open `data/synthetic/pairs.csv`
and read a few rows. Then delete the trial file and do the full run:
```
rm data/synthetic/pairs_raw.csv
python src/generate.py
```
If it crashes or you stop it, run the same command again: it skips what's done.
If you hit rate limits, add `--workers 4`.

## 2. Filter (T7), 2 min + manual check

```
python src/filter_pairs.py
```
Numbers go in `results/filter_stats.txt`. Put `results/manual_review_sample.csv` into a Google
Sheet; each person fills ~33 rows (`decision` = keep/reject, `reason` = meaning changed / not
casual / unnatural / other). Write the totals into the last line of `filter_stats.txt`.
**Don't wait on the manual check to keep going** — it's for the slide, not a blocker.

## 3. Split (T8), seconds

```
python src/make_splits.py
```

## 4. Baseline (T10), ~10–15 min

```
python src/baseline.py --limit 10      # look at the 3 printed examples
python src/baseline.py                 # full test set
```
Uses `gemini-2.5-flash` at temperature 0. To use another Gemini model, set
`GEMINI_MODEL=...` in `.env`.

## 5. Metrics (T11), ~5 min

```
python src/evaluate.py
```
If BERTScore/PyTorch won't install or is too slow, `python src/evaluate.py --no-bertscore`.
Open `results/spain_forms_hits.csv` and confirm each hit is real.

## 6. Error analysis (T12), ~30 min of reading

```
python src/error_analysis.py sample
```
Label each of the 30 rows in `results/error_analysis.csv` (one category each:
`good, register_collapse, regional_flattening, meaning_drift, over_embellishment, missed_error`),
and mark 3 `strong` + 3 `weak` in the `pick` column. Then:
```
python src/error_analysis.py count
```
→ `results/error_categories.png` (slide 9 chart) and `results/examples_for_slides.txt`.

## Running out of time?

The plan's fallback is fine: `python src/generate.py --limit 1200` gives ~1,000 pairs after
filtering and a test set of ~100. Run steps 2–6 the same way. Say in the talk the full set is
in progress.

## What goes on the slides
- Slide 5: `results/data_stats.txt`, `filter_stats.txt`, `split_stats.txt`
- Slide 8: `results/metrics.csv`
- Slide 9: `error_categories.png`, `examples_for_slides.txt`
