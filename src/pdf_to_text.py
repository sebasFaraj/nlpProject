"""
WRITTEN BY CLAUDE

Usage (from the repo root):
    python src/pdf_to_text.py

Reads:   data/pdfs/<author>_<work>.pdf
Writes:  data/raw/<author>_<work>.txt

What it does to each PDF:
  1. Pulls the text out page by page, in reading order.
  2. Removes running headers/footers: short lines (like the book title or
     author name) that repeat on many pages, plus lines that are just a page
     number.
  3. Joins the pages back together and saves the result as UTF-8.

It does NOT remove front matter (title page, prologue, table of contents).
Do that by hand afterward, as described in T3. Line-wrapping and hyphenation
get fixed later in T4.
"""

import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf  # pip install pymupdf

PDF_DIR = Path("data/pdfs")
OUT_DIR = Path("data/raw")

# A short line that shows up on at least this fraction of pages is treated as
# a running header/footer and removed.
REPEAT_THRESHOLD = 0.3
MAX_HEADER_LEN = 60  # running headers are short; never delete long lines

PAGE_NUMBER = re.compile(r"^\s*(\d{1,4}|[ivxlcdm]{1,6})\s*$", re.IGNORECASE)


def header_key(line):
    """Normalize a line so the same header matches even with different page numbers."""
    return re.sub(r"\d+", "#", line.strip().lower())


def extract_pages(pdf_path):
    with pymupdf.open(pdf_path) as doc:
        return [page.get_text("text", sort=True) for page in doc]


def find_running_headers(pages):
    """Return the set of short lines that repeat across many pages."""
    counts = Counter()
    for text in pages:
        # Count each distinct line once per page.
        keys = {header_key(l) for l in text.splitlines() if 0 < len(l.strip()) <= MAX_HEADER_LEN}
        counts.update(keys)
    min_pages = max(3, int(len(pages) * REPEAT_THRESHOLD))
    return {key for key, n in counts.items() if n >= min_pages}


def clean_page(text, headers):
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if PAGE_NUMBER.match(stripped):
            continue
        if stripped and len(stripped) <= MAX_HEADER_LEN and header_key(stripped) in headers:
            continue
        kept.append(line.rstrip())
    return "\n".join(kept).strip()


def convert(pdf_path, out_path):
    pages = extract_pages(pdf_path)
    headers = find_running_headers(pages)
    cleaned = [clean_page(p, headers) for p in pages]

    # Pages join with a single newline: a sentence that runs across a page
    # break stays one paragraph (T4 turns single newlines into spaces).
    text = "\n".join(p for p in cleaned if p)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse big gaps

    out_path.write_text(text, encoding="utf-8")

    chars_per_page = len(text) / max(len(pages), 1)
    warning = "  <-- very little text: may be a scanned PDF, check it" if chars_per_page < 200 else ""
    print(f"{pdf_path.name}: {len(pages)} pages, {len(text):,} chars, "
          f"{len(headers)} header pattern(s) removed{warning}")


def main():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}/. Put the book PDFs there first.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pdf_path in pdfs:
        out_path = OUT_DIR / (pdf_path.stem + ".txt")
        try:
            convert(pdf_path, out_path)
        except Exception as e:
            print(f"[FAIL] {pdf_path.name}: {type(e).__name__}: {e}")

    print(f"\nDone. Text files are in {OUT_DIR}/")


if __name__ == "__main__":
    main()