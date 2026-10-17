# Spanish Literary Style Transfer

Rewriting casual, error-prone Spanish as literary South American prose.

No parallel corpus exists for this task, so we built one synthetically. We extracted 4,200
sentences from seven works by five South American authors, had GPT-5-mini rewrite each one as
casual Spanish, and added controlled typing errors to about half of them. Each pair uses the
casual sentence as input and the original literary sentence as the target. After filtering we
kept **3,186 pairs**, split 80/10/10 into train/val/test. We scored a Gemini baseline
(zero-shot and few-shot) on the test set.

## Data

| Author | Country | Work | Year |
|---|---|---|---|
| Gabriel García Márquez | Colombia | Cien años de soledad | 1967 |
| Gabriel García Márquez | Colombia | Crónica de una muerte anunciada | 1981 |
| Isabel Allende | Chile | La casa de los espíritus | 1982 |
| Isabel Allende | Chile | La ciudad de las bestias | 2002 |
| Jorge Luis Borges | Argentina | Ficciones | 1944 |
| Mario Vargas Llosa | Peru | La ciudad y los perros | 1963 |
| Horacio Quiroga | Uruguay | Cuentos de amor de locura y de muerte | 1917 |

We sampled 600 sentences per work (8–40 words; dialogue, fragments and lines with digits
removed). See `data/sources.csv` and `results/data_stats.txt`.

| Stage | Pairs |
|---|---|
| Generated | 4,200 |
| Removed: casual too similar to literary (>0.85) | 934 |
| Removed: length ratio outside 30–150% | 80 |
| **Kept** | **3,186** (75.9%) |
| Train / val / test | 2,548 / 318 / 320 |

Sentences from the same book appear in every split. See `results/filter_stats.txt` and
`results/split_stats.txt`.

## Results

Test set (320 pairs). The baseline was Gemini at temperature 0, with 5 few-shot examples drawn
from train. The exact model and prompts are in `results/baseline_log.txt`.

| System | BLEU | ROUGE-L | BERTScore-F1 | Avg. words (ref. 20.4) |
|---|---|---|---|---|
| Copy input | 33.92 | 0.594 | 0.866 | 21.1 |
| Zero-shot | 23.17 | 0.505 | 0.836 | 23.6 |
| Few-shot | 35.37 | 0.597 | 0.871 | 21.4 |

None of the systems produced Spain-only forms (*vosotros*, *habéis*, …). Copying the input
already scores close to few-shot, because there is only one reference per sentence and many
valid literary rewrites. This is why human evaluation matters for this task.

Error analysis of 30 few-shot outputs (`results/error_categories.png`): good 15,
over-embellishment 7, register collapse 4, regional flattening 2, meaning drift 2, missed
error 0.

## Repository layout

```
data/
  raw/              plain text of each book (from pdf_to_text.py)
  clean/            literary_sentences.csv: the 4,200 sampled sentences
  synthetic/        pairs.csv (all pairs + errors), pairs_filtered.csv (after filtering)
    intermediate/   pairs_raw.csv (casual rewrites before errors), pilot_pairs.csv (prompt pilot)
  splits/           train.csv, val.csv, test.csv
  sources.csv       the works used
src/                pipeline scripts (below); prompts/generation_prompt.txt
results/            stats, baseline outputs, metrics, error analysis
```

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

Create a `.env` file in the repo root:

```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
# optional overrides: OPENAI_MODEL (default gpt-5-mini), GEMINI_MODEL (default gemini-3.8-flash)
```

Check it with `python src/test_api.py`.

## Reproducing the pipeline

Run every command from the repo root. The generation and baseline steps call paid APIs. Both
resume where they left off if interrupted.

```
# 1. PDFs in data/pdfs/ -> data/raw/*.txt (remove front matter by hand afterwards)
python src/pdf_to_text.py

# 2. Extract and sample sentences, once per book
python src/extract_sentences.py data/raw/borges_ficciones.txt \
    --author "Jorge Luis Borges" --country Argentina --work "Ficciones"

# 3. Pilot the generation prompt on 20 sentences (optional)
python src/pilot.py

# 4. Casual rewrites with GPT-5-mini, then add errors (add_errors.py runs automatically)
python src/generate.py

# 5. Filter, 6. split
python src/filter_pairs.py
python src/make_splits.py

# 7. Baseline: zero-shot, few-shot and copy-input on the test set
python src/baseline.py

# 8. Metrics (add --no-bertscore to skip BERTScore)
python src/evaluate.py

# 9. Error analysis: sample 30 outputs, label them by hand, then count
python src/error_analysis.py sample
python src/error_analysis.py count
```

## Limitations

- The casual inputs are LLM-generated, so they may not match real casual writing.
- There is one reference per input, which keeps the automatic metrics low.
- Every book appears in every split. A held-out-book test is still to do.
