# nlpProject

Instructions: https://claude.ai/artifact/9rE4BkWd2Tb7Kd5VaCGh9r#c16260ee-7815

PM2 Task Plan: Spanish Literary Style Transfer
Oct 8, 2026 · @Someone
Overview
By Monday Oct 12 we need a finished dataset of at least 3,000 (casual → literary) Spanish sentence pairs, a baseline run on it with scores, and a 7-minute talk. Slides go to the instructor by 11:00 a.m. Monday. The written report is due Oct 16 and is not covered here.
We are following the professor's Direction 1: keep style transfer and build the data synthetically.
1. Collect about 3,500 real literary sentences from South American authors.
2. Have an LLM (GPT-5-mini) rewrite each one as casual Spanish, with some typical errors added.
3. That gives pairs: casual sentence = input, original literary sentence = expected output.
4. Run the baseline (an LLM asked to do the casual → literary rewrite) on the test pairs and score it.
How to use this doc: each task below has a goal, step-by-step instructions, what to hand off, and a "done when" check. Put your name in the Owner column of the task table and update Status as you go. Tasks marked depends on can't start until the earlier task is finished.
Task summary
Fourteen tasks, about 35–40 hours of work total. The critical path is T3 → T4 → T6 → T7 → T8 → T10 → T11 → T13: delays there push everything back. T1, T2, T5 and T9 can start right away in parallel.
#
Task
Depends on
Effort
Owner
Status
T1
Set up shared repo and Python environment
—
1–2 h

Not started
T2
Pick authors and works
—
1 h

Not started
T3
Collect raw literary texts
T2
3–4 h

Not started
T4
Clean texts and extract sentences
T1, T3
3–4 h

Not started
T5
Write and pilot the generation prompt
T1
2–3 h

Not started
T6
Run full synthetic generation
T4, T5
2 h

Not started
T7
Quality-check and filter pairs
T6
2–3 h

Not started
T8
Make train/val/test splits
T7
1 h

Not started
T9
Write real-world casual sentences (optional before Monday)
—
1 h per person
All
Not started
T10
Run the baseline (zero-shot and few-shot)
T8
3 h

Not started
T11
Compute automatic metrics
T10
2 h

Not started
T12
Pick examples and do error analysis
T10
2–3 h

Not started
T13
Build the slides
T11, T12
4–5 h

Not started
T14
Rehearse and prep Q&A
T13
1–2 h
All
Not started
Phase 0: Setup
T1. Set up shared repo and Python environment
Goal: everyone can run the same code with the same libraries and API keys.
1. Create a private GitHub repo and add all three of us.
2. Create these folders: data/raw, data/clean, data/synthetic, data/splits, src, results.
3. Add a requirements.txt with: openai, google-genai, pandas, spacy, sacrebleu, rouge-score, bert-score, tqdm, python-dotenv.
4. Install with pip install -r requirements.txt, then download the Spanish sentence splitter: python -m spacy download es_core_news_sm.
5. Get an OpenAI API key and a Gemini API key. Put them in a file named .env (one per line: OPENAI_API_KEY=..., GEMINI_API_KEY=...).
6. Add .env to .gitignore. Never commit API keys. Share them privately.
7. Write src/test_api.py: send "Hola, ¿cómo estás?" to both APIs and print the replies.
8. Add a short README.md (one paragraph on the project for now; it gets finished for the report).
Hand off: repo link + .env values shared privately.
Done when: all three of us can clone the repo, install, and run test_api.py successfully.
T2. Pick authors and works
Goal: a fixed list of source books, so collection (T3) can start.
1. Choose 8–10 prose works from at least 5 authors and at least 4 countries, so the data isn't one writer's style.
2. Prose only. Skip poetry and works written in heavy rural dialect (for example Martín Fierro).
3. Starting list to pick from:
    ◦ Gabriel García Márquez (Colombia): Cien años de soledad, Crónica de una muerte anunciada
    ◦ Jorge Luis Borges (Argentina): Ficciones
    ◦ Julio Cortázar (Argentina): Bestiario or Rayuela
    ◦ Mario Vargas Llosa (Peru): La ciudad y los perros
    ◦ Isabel Allende (Chile): La casa de los espíritus
    ◦ Horacio Quiroga (Uruguay): Cuentos de amor de locura y de muerte
    ◦ Ricardo Palma (Peru): Tradiciones peruanas
    ◦ Jorge Isaacs (Colombia): María
    ◦ Rómulo Gallegos (Venezuela): Doña Bárbara
4. Plan to take at most about 400 sentences per work, so no single book dominates.
5. Create data/sources.csv with columns: author, country, work, year, source (where the text came from).
6. Write one sentence for the report defining our scope, e.g. "literary prose by 19th–20th century South American authors."
Hand off: data/sources.csv.
Done when: all three agree on the list.
Phase 1: Data
T3. Collect raw literary texts
Depends on T2.
Goal: one clean plain-text file per book.
1. For each work in data/sources.csv, get the full text. Try Project Gutenberg first (download the "Plain Text UTF-8" version), then the Biblioteca Virtual Miguel de Cervantes, then any other copy.
2. Save each as data/raw/<author>_<work>.txt, for example data/raw/quiroga_cuentos.txt.
3. Open each file and delete by hand everything that isn't the story: title pages, prefaces, table of contents, footnotes, and the Project Gutenberg license text at the top and bottom.
4. Check the accents display correctly (á, é, ñ, ¿, ¡). If you see junk like "Ã¡", the encoding is wrong: re-save the file as UTF-8.
5. Fill in the source column in sources.csv with where each text came from.
Hand off: the files in data/raw/.
Done when: every work has a file that starts at the first chapter and shows accents correctly.
T4. Clean texts and extract sentences
Depends on T1 and T3.
Goal: a table of about 3,500 good literary sentences. Write this as src/extract_sentences.py.
1. Read each raw file and normalize the characters (Python: unicodedata.normalize("NFC", text)).
2. Fix line wrapping: a single line break inside a paragraph becomes a space; a blank line means a new paragraph. Rejoin words split by a hyphen at a line end.
3. Split each paragraph into sentences with spaCy's Spanish model (es_core_news_sm). Process paragraph by paragraph so long books don't hit spaCy's length limit.
4. Keep only sentences that:
    ◦ have 8–40 words
    ◦ start with a capital letter, ¿ or ¡
    ◦ end with . ? ! or …
    ◦ do not start with a dialogue dash (— or –); we want narration, not dialogue
    ◦ have no digits and aren't in ALL CAPS (these are usually headings)
5. Remove exact duplicates.
6. Randomly sample up to 400 sentences per work (use random_state=42 so results are reproducible). If the total is under 3,500, raise the cap for the longer books.
7. Save data/clean/literary_sentences.csv with columns: id, author, country, work, sentence.
8. Print and save stats to results/data_stats.txt: sentences per work and per author, average words per sentence. These go on a slide.
9. Read 30 random sentences by eye. If junk got through (headings, broken sentences), tighten the filters and rerun.
Hand off: literary_sentences.csv + data_stats.txt.
Done when: at least 3,500 sentences and the 30-sentence spot check looks clean.
T5. Write and pilot the generation prompt
Depends on T1. Can start before T4 finishes: hand-copy 20 sentences from any of the books to test with.
Goal: a prompt that reliably turns a literary sentence into a natural casual one with the same meaning.
Design decision: the LLM only changes the style. A separate script adds grammar errors afterward. That keeps the errors controlled and easy to describe in the report (we can state the exact error rate), instead of whatever errors the LLM decides to make.
1. Start from this prompt (instructions in English are fine; the output will be Spanish):
   You are helping build a dataset. You will receive one sentence from a Spanish-language novel. Rewrite it the way an ordinary person would say the same thing in a casual text message or a quick informal email. Rules: keep the same meaning and every key fact (who, what, where, when); do not add or remove information. Replace literary vocabulary with everyday words. Simplify the sentence structure. Use natural colloquial expressions where they fit, without overdoing slang. Keep names unchanged. Write in Spanish. Return only the rewritten sentence, with no quotes or explanation.
2. Run it with GPT-5-mini on the 20 test sentences. Save the exact model name you used.
3. All three of us review the 20 outputs. A pair is good if the meaning is the same, it clearly sounds more casual, and it sounds like something a real person would write.
4. Target: at least 17 of 20 good. If not, adjust the rules based on what went wrong and rerun.
5. Save the final prompt as src/prompts/generation_prompt.txt.
6. Write src/add_errors.py. For each casual sentence, with 50% probability, apply 1–2 of these at random (seed 42):
    ◦ remove accents from 1–3 words (é → e, á → a, ñ stays)
    ◦ make the first letter lowercase
    ◦ drop the opening ¿ or ¡
    ◦ drop the final period
7. Keep both versions: casual_clean (no added errors) and casual_input (with errors). The model's input is casual_input.
Hand off: final prompt file + add_errors.py + the 20 pilot pairs (good slide material).
Done when: 17/20 pilot pairs pass review.
T6. Run full synthetic generation
Depends on T4 and T5.
Goal: a casual version of every literary sentence. Write this as src/generate.py.
1. First run on 50 sentences only. Check the API usage page to see what it cost, and multiply out to estimate the full run.
2. Then loop over all of literary_sentences.csv, one API call per sentence. Running 5–10 calls in parallel (threads) makes it much faster.
3. Save progress to data/synthetic/pairs_raw.csv every 100 sentences. On restart, skip ids already done, so a crash doesn't lose work.
4. If a call fails, wait a few seconds and retry, up to 3 times.
5. Write the model name, date, and prompt file used to results/generation_log.txt.
6. Run add_errors.py on the result to make data/synthetic/pairs.csv with columns: id, author, work, literary, casual_clean, casual_input, has_errors.
Hand off: pairs.csv + generation_log.txt.
Done when: every sentence has a casual version.
T7. Quality-check and filter pairs
Depends on T6.
Goal: remove bad pairs and record how many we removed and why (the report needs these numbers).
1. Write src/filter_pairs.py. Drop a pair automatically if:
    ◦ the casual version is empty
    ◦ the casual version is nearly identical to the literary one (Python's difflib.SequenceMatcher ratio above 0.85)
    ◦ the casual version is under 30% or over 150% of the literary sentence's word count
    ◦ the LLM added commentary (contains phrases like "Aquí tienes" or quotation marks around the whole sentence)
2. Log how many pairs each rule removed.
3. Manual check: pick 100 random remaining pairs and split them between us (about 33 each). In a shared sheet, mark each one keep or reject, with a reason: meaning changed, not casual, unnatural, or other.
4. If more than 15% get rejected, find the pattern, fix the prompt, and regenerate just the bad ones.
5. Record for the report: count before filtering, count after, and the reject rate by reason.
Hand off: data/synthetic/pairs_filtered.csv + the numbers.
Done when: at least 3,000 pairs remain and the numbers are written down.
T8. Make train/val/test splits
Depends on T7.
Goal: separate data for training, tuning, and final testing. The baseline is scored on test only.
1. Shuffle pairs_filtered.csv with seed 42 and split 80% train, 10% validation, 10% test.
2. Save data/splits/train.csv, val.csv, test.csv.
3. Add a check in code that no id appears in more than one split.
4. Save stats to results/split_stats.txt: number of pairs per split, average words per input and output, sentences per author in each split.
5. Note for the report's limitations: sentences from the same book appear in every split, so the model may partly learn each author's habits. Holding out whole books is a stricter version we can try later.
Hand off: the three split files + stats.
Done when: the files exist and the overlap check passes.
T9. Write real-world casual sentences (optional before Monday)
No dependencies. Everyone does this.
Goal: a small test set of genuinely human casual Spanish, to check the system works on real writing and not just LLM-made inputs.
1. Each person writes 30–50 sentences the way they'd actually write: texts to friends, quick work emails, notes to yourself.
2. Mix topics. Leave in natural typos and missing accents. Don't look at the literary data first.
3. Save as data/real_world/<your name>.txt, one sentence per line.
4. No literary "answer" is needed for these. They're for human evaluation and examples.
Done when: about 100–150 sentences total. Required before the report on Oct 16, not before the talk.
Phase 2: Baseline
T10. Run the baseline (zero-shot and few-shot)
Depends on T8.
Goal: outputs from a strong existing approach on every test pair. Write this as src/baseline.py.
1. Use a Gemini model, not GPT. GPT made our inputs, so using a different model family keeps the comparison fair. Write down the exact model name.
2. Set temperature to 0 so results are as repeatable as possible.
3. Zero-shot prompt (instructions only, no examples):
   Rewrite the following casual Spanish text as polished literary Spanish, in the style of South American literary prose. Correct any spelling, accent, or grammar errors. Keep exactly the same meaning. Return only the rewritten text.
4. Few-shot prompt: the same instructions plus 5 example pairs (casual_input → literary) picked at random from train.csv with seed 42. Use the same 5 examples for every test sentence.
5. Test on 10 sentences first, then run both prompts on the full test set.
6. Also make a copy-input file where the "output" is just the input unchanged. It isn't a real baseline; it shows the score you get for doing nothing, so the other numbers have something to compare against.
7. Save results/baseline_zeroshot.csv, results/baseline_fewshot.csv, results/copy_input.csv, each with columns: id, casual_input, literary_reference, output.
8. Write model name, date, temperature and both prompts to results/baseline_log.txt. The README needs these so someone can rerun it.
Hand off: the three output files + log.
Done when: both baseline files have an output for every test pair.
T11. Compute automatic metrics
Depends on T10.
Goal: one results table. Write this as src/evaluate.py. Each metric compares the output to the original literary sentence (the "reference").
• BLEU (sacrebleu): counts how many 1–4 word sequences in the output also appear in the reference. Scale 0–100. Strict: a synonym counts as wrong.
• ROUGE-L (rouge-score): length of the longest run of matching words, in order, between output and reference. Scale 0–1.
• BERTScore (bert-score, with lang="es"): compares meaning using a language model, so synonyms get partial credit. Report the F1 value.
1. Compute all three for copy-input, zero-shot and few-shot.
2. Watch out: rouge-score's default tokenizer deletes characters outside a–z, which mangles Spanish words with accents or ñ. Pass a simple whitespace tokenizer instead. Sanity check: a sentence scored against itself must give 1.0, and "canción" vs "cancion" must not count as a match.
3. Also report average output length in words. A model that writes much longer outputs is padding, which matters for the analysis.
4. Quick regional check: count how many outputs use Spain-only forms (vosotros, vosotras, habéis, tenéis, sois, estáis). Look at each hit by eye to confirm it's real.
5. Save results/metrics.csv: one row per system, one column per metric.
Scores will be fairly low even for good rewrites, because there are many valid literary versions of a sentence and we only have one reference. That's expected and worth saying in the talk; it's why human evaluation is our main metric later.
Hand off: metrics.csv.
Done when: the table has all three systems and all metrics.
T12. Pick examples and do error analysis
Depends on T10. Can run alongside T11.
Goal: show where the baseline succeeds and fails. This is what motivates our fine-tuned model.
1. Take 30 random test outputs from the few-shot run.
2. Label each output with one main category:
    ◦ Good: literary, correct, same meaning
    ◦ Register collapse: formal but flat or bureaucratic, not literary
    ◦ Regional flattening: Spain-style or dialect-neutral wording
    ◦ Meaning drift: changed or lost information
    ◦ Over-embellishment: adds flowery details that weren't in the input
    ◦ Missed error: left an accent or grammar error unfixed
3. Count how many fall in each category (this becomes a small bar chart on a slide).
4. Choose 3 strong and 3 weak examples. For each, keep the input, the baseline output and the reference together.
5. Write 3–4 sentences: what the baseline does well, where it fails, and which failures our fine-tuned model should fix.
Hand off: category counts + 6 examples + the short write-up.
Done when: all of the above is in the shared folder.
Phase 3: Presentation
T13. Build the slides
Depends on T11 and T12 for the results slides. Slides 1–4, 6, 7 and 10 can be built now.
Goal: about 11 slides that fit in 7 minutes (roughly 40 seconds each).
Slide
Content
Time
1
Title and team
10 s
2
Motivation: Martin's novel, Sebastian's client emails; LLMs give bland, generic "formal" Spanish
30 s
3
Task: casual → literary South American Spanish, fixing errors along the way. One input/output example
45 s
4
Data problem and fix: no paired data exists, so we built it. Pipeline: literary sentence → GPT casual rewrite → add errors → pair
45 s
5
Data stats: authors and countries, pair counts, split sizes, filter/reject rate, 2 example pairs
45 s
6
Baseline: Gemini zero-shot and few-shot, temperature 0, different model family from the data generator
30 s
7
Metrics: BLEU, ROUGE-L, BERTScore, plus human evaluation later; why one reference keeps scores low
30 s
8
Results table from T11
45 s
9
Examples: 1 strong, 2 weak, plus the error-category bar chart from T12
60 s
10
Next steps: fine-tune an encoder-decoder (mT5 or mBART) and a decoder-only model (Qwen3) on our train set; human eval on the real-world set
45 s
11
Limitations and what feedback we want from the class
20 s
For slide 10, note that the original BART, T5 and GPT-2 were trained mostly on English, so we use their multilingual versions. That shows we checked.
Hand off: the deck, emailed to the instructor before 11:00 a.m. Monday Oct 12 (one named person owns sending it).
Done when: all slides filled with real numbers and examples.
T14. Rehearse and prep Q&A
Depends on T13. Everyone.
1. Decide who presents which slides.
2. Do two full timed run-throughs. Must finish under 7:00; cut words, not slides.
3. Prepare short answers to the questions we're most likely to get:
    ◦ Isn't LLM-made casual text different from real casual writing? Yes, which is why we built a real-world test set (T9) for human evaluation.
    ◦ Why not just use ChatGPT? That's our baseline. A fine-tuned smaller model is cheaper, more controllable, and can be specialized to South American style.
    ◦ How do you know outputs sound South American? The regional-form check plus human evaluation; we name it as a limitation.
    ◦ Why are BLEU scores so low? One reference per sentence; many valid rewrites score low.
    ◦ Could the model just memorize authors? Same books appear in every split; a held-out-book test is planned.
Done when: two clean timed runs and everyone can answer the questions above.
Schedule and scope
The data must be done by Saturday night, or there are no results to present Sunday.
Day
Work on
Finished by end of day
Thu Oct 8
T1, T2; start T3
Repo works, author list agreed
Fri Oct 9
T3, T4, T5
3,500 clean sentences; prompt passes the 20-sentence pilot
Sat Oct 10
T6, T7, T8; start T10
Filtered pairs split into train/val/test
Sun Oct 11
T10, T11, T12, T13; evening rehearsal
Baseline scored, slides filled
Mon Oct 12
T14 final run
Slides emailed before 11:00 a.m.
If we fall behind: the talk needs real numbers, not the full dataset. Fallback is about 1,000 pairs, a test set of 100–200, and the baseline scored on that. Say in the talk that the full 3,000+ set is in progress, and finish it before the Oct 16 report.
Not needed before Monday:
• Fine-tuning any model (that's PM3/PM4)
• The full human evaluation
• The real-world sentence set (T9)
• The finished README and code zip
• The written report (due Oct 16)
