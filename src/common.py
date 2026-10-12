"""Shared helpers: API clients with retries, and a parallel runner that saves progress."""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def with_retries(fn, tries=4, wait=3):
    """Call fn(); on an error wait a few seconds and retry (wait doubles each time)."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if attempt == tries - 1:
                raise
            print(f"  retry {attempt + 1} after error: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(wait * 2 ** attempt)


# ---------- OpenAI (data generation) ----------
_openai = None

def call_openai(instructions, text):
    global _openai
    if _openai is None:
        from openai import OpenAI
        _openai = OpenAI()

    def go():
        kwargs = dict(model=OPENAI_MODEL, instructions=instructions, input=text)
        if OPENAI_MODEL.startswith("gpt-5"):
            # GPT-5 models are reasoning models; "minimal" makes them fast and cheap.
            kwargs["reasoning"] = {"effort": "minimal"}
        return _openai.responses.create(**kwargs).output_text.strip()

    return with_retries(go)


# ---------- Gemini (baseline) ----------
_gemini = None
_gemini_lock = threading.Lock()
_gemini_no_thinking = True

def call_gemini(system, text, temperature=0.0):
    global _gemini, _gemini_no_thinking
    from google.genai import types
    if _gemini is None:
        # Lock so parallel workers share one client; extra clients get closed mid-request.
        with _gemini_lock:
            if _gemini is None:
                from google import genai
                _gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def go():
        global _gemini_no_thinking
        cfg = dict(system_instruction=system, temperature=temperature)
        if _gemini_no_thinking:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        try:
            r = _gemini.models.generate_content(
                model=GEMINI_MODEL, contents=text, config=types.GenerateContentConfig(**cfg))
        except Exception as e:
            # Some models don't allow turning thinking off; drop that setting and retry.
            if _gemini_no_thinking and "thinking" in str(e).lower():
                _gemini_no_thinking = False
                return go()
            raise
        return (r.text or "").strip()

    return with_retries(go)


# ---------- parallel loop with checkpointing ----------
def run_parallel(rows, work_fn, out_path, workers=8, save_every=100):
    """
    rows: list of dicts, each with an "id". work_fn(row) -> dict of new columns.
    Results are appended to out_path every `save_every` rows; ids already in
    out_path are skipped, so you can stop and rerun without losing work.
    """
    out_path = Path(out_path)
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    done_ids = set(done["id"]) if len(done) else set()
    todo = [r for r in rows if r["id"] not in done_ids]
    print(f"{len(done_ids)} already done, {len(todo)} to go -> {out_path}")

    buffer, failed = [], 0

    def flush():
        nonlocal done, buffer
        if buffer:
            done = pd.concat([done, pd.DataFrame(buffer)], ignore_index=True)
            done.to_csv(out_path, index=False)
            buffer = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work_fn, r): r for r in todo}
        for f in tqdm(as_completed(futures), total=len(futures)):
            r = futures[f]
            try:
                buffer.append({**r, **f.result()})
            except Exception as e:
                failed += 1
                print(f"  FAILED {r['id']}: {type(e).__name__}: {str(e)[:150]}")
            if len(buffer) >= save_every:
                flush()
    flush()
    if failed:
        print(f"{failed} rows failed. Just rerun the same command to retry only those.")
    return done
