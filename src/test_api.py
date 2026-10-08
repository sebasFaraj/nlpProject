"""
T1 environment check.

Confirms that:
  1. the .env file is found and both API keys are set,
  2. the OpenAI API answers (used for synthetic data generation),
  3. the Gemini API answers (used for the baseline),
  4. spaCy's Spanish model is installed (used for sentence splitting).

Run from the repo root:  python src/test_api.py
"""

import os
import sys

from dotenv import load_dotenv

TEST_SENTENCE = "Hola, ¿cómo estás? Responde en una sola frase corta en español."


def check_keys():
    """Load .env and make sure both keys exist before calling anything."""
    load_dotenv()  # reads .env from the current folder (the repo root)
    missing = [k for k in ("OPENAI_API_KEY", "GEMINI_API_KEY") if not os.getenv(k)]
    if missing:
        print(f"[FAIL] Missing in .env: {', '.join(missing)}")
        print("       Copy .env.example to .env and fill in the keys.")
        return False
    print("[ OK ] Both API keys found in .env")
    return True


def test_openai():
    from openai import OpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    client = OpenAI()  # picks up OPENAI_API_KEY automatically
    response = client.responses.create(model=model, input=TEST_SENTENCE)
    print(f"[ OK ] OpenAI ({model}) replied: {response.output_text.strip()}")


def test_gemini():
    from google import genai

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(model=model, contents=TEST_SENTENCE)
    print(f"[ OK ] Gemini ({model}) replied: {response.text.strip()}")


def test_spacy():
    import spacy

    nlp = spacy.load("es_core_news_sm")
    doc = nlp("Ayer amaneció enfermo el muchacho. No podrá acompañarnos a la fiesta.")
    sentences = [s.text for s in doc.sents]
    print(f"[ OK ] spaCy Spanish model split {len(sentences)} sentences: {sentences}")


def main():
    if not check_keys():
        sys.exit(1)

    # Run each check separately so one failure doesn't hide the others.
    failures = 0
    for name, test in [("OpenAI", test_openai), ("Gemini", test_gemini), ("spaCy", test_spacy)]:
        try:
            test()
        except Exception as e:
            failures += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")

    print()
    if failures:
        print(f"{failures} check(s) failed. Fix those before moving on to T4/T5.")
        sys.exit(1)
    print("All checks passed. Environment is ready.")


if __name__ == "__main__":
    main()