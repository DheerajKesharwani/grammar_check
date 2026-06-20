"""
Grammar & Redundancy checker for PDF files  -- PURE PYTHON, NO JAVA.
No LLM, no sentence-transformers, no LanguageTool.

Grammar  : pyspellchecker (bundled dictionary) + hand-written rules.
           Optional spaCy POS checks turn on automatically IF the model
           'en_core_web_sm' is installed (otherwise silently skipped).
Redundancy: pleonasm dictionary + overused-word counts + TF-IDF cosine.

Setup:
    pip install pypdf scikit-learn pyspellchecker
    # optional, for extra grammar checks:
    #   pip install spacy && python -m spacy download en_core_web_sm

Usage:
    python check_text_pure.py yourfile.pdf
"""

import sys
import re
from collections import Counter

from pypdf import PdfReader
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------- dictionaries

PLEONASMS = {
    r"\batm machine\b": "ATM",
    r"\bpin number\b": "PIN",
    r"\bfree gift\b": "gift",
    r"\bend result\b": "result",
    r"\bfinal outcome\b": "outcome",
    r"\bpast history\b": "history",
    r"\bin order to\b": "to",
    r"\bdue to the fact that\b": "because",
    r"\bat this point in time\b": "now",
    r"\beach and every\b": "each / every",
    r"\bbasic fundamentals\b": "fundamentals",
    r"\babsolutely essential\b": "essential",
    r"\bcompletely eliminate\b": "eliminate",
    r"\bclose proximity\b": "near",
    r"\bnew innovation\b": "innovation",
    r"\brepeat again\b": "repeat",
    r"\bunexpected surprise\b": "surprise",
    r"\bfuture plans\b": "plans",
}

# Common missing-apostrophe contractions (spellcheckers handle these poorly).
CONTRACTIONS = {
    "dont": "don't", "cant": "can't", "wont": "won't", "didnt": "didn't",
    "doesnt": "doesn't", "isnt": "isn't", "arent": "aren't", "wasnt": "wasn't",
    "werent": "weren't", "couldnt": "couldn't", "shouldnt": "shouldn't",
    "wouldnt": "wouldn't", "hasnt": "hasn't", "havent": "haven't",
    "hadnt": "hadn't", "im": "I'm", "ive": "I've", "youre": "you're",
    "theyre": "they're", "weve": "we've", "thats": "that's",
}

# Negative triggers for crude double-negative detection.
NEG_WORDS = {"no", "not", "none", "nothing", "nobody", "never",
             "nowhere", "neither", "n't"}
NEG_FOLLOW = {"no", "none", "nothing", "nobody", "never", "nowhere", "neither"}

# 'an' is correct before these despite starting with a consonant letter.
AN_EXCEPTIONS = {"hour", "honest", "honor", "honour", "heir"}
# 'a' is correct before these despite starting with a vowel letter.
A_EXCEPTIONS = {"university", "unicorn", "european", "one", "user", "unit",
                "useful", "unique", "united"}


# ------------------------------------------------------------------- pdf + text

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    chunks = [(p.extract_text() or "") for p in reader.pages]
    text = "\n".join(chunks)
    text = re.sub(r"-\n", "", text)        # de-hyphenate split words
    text = re.sub(r"\s*\n\s*", " ", text)  # join lines
    text = re.sub(r"\s{2,}", " ", text)    # squeeze spaces
    return text.strip()


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


# --------------------------------------------------------------------- grammar

def check_spelling(text):
    spell = SpellChecker()
    issues = []
    seen = set()
    # words: letters and inner apostrophes
    for w in re.findall(r"\b[A-Za-z][A-Za-z']+\b", text):
        low = w.lower()
        if low in seen:
            continue
        # skip likely proper nouns (capitalised, not at sentence start we
        # can't easily know, so just skip all-caps and Capitalised words)
        if w[0].isupper():
            continue
        if low in spell:
            continue
        seen.add(low)
        corrections = spell.candidates(low)
        suggestion = spell.correction(low)
        issues.append({
            "type": "spelling",
            "bad": w,
            "suggestion": suggestion or "(no suggestion)",
        })
    return issues


def check_rules(text, sentences):
    issues = []

    # 1. missing-apostrophe contractions
    for m in re.finditer(r"\b([A-Za-z]+)\b", text):
        low = m.group(1).lower()
        if low in CONTRACTIONS:
            issues.append({
                "type": "contraction",
                "bad": m.group(1),
                "suggestion": CONTRACTIONS[low],
            })

    # 2. doubled words ("the the")
    for m in re.finditer(r"\b(\w+)\s+\1\b", text, flags=re.IGNORECASE):
        issues.append({
            "type": "doubled word",
            "bad": m.group(0),
            "suggestion": m.group(1),
        })

    # 3. a / an misuse
    for m in re.finditer(r"\b(a|an)\s+([A-Za-z]+)", text, flags=re.IGNORECASE):
        art, nxt = m.group(1).lower(), m.group(2).lower()
        starts_vowel = nxt[0] in "aeiou"
        if nxt in AN_EXCEPTIONS:
            starts_vowel = True
        elif nxt in A_EXCEPTIONS:
            starts_vowel = False
        correct = "an" if starts_vowel else "a"
        if art != correct:
            issues.append({
                "type": "a/an",
                "bad": m.group(0),
                "suggestion": f"{correct} {m.group(2)}",
            })

    # 4. crude double negatives (per sentence)
    # negative-verb typos like "dont"/"cant" also count as a negative signal
    neg_typos = {k for k, v in CONTRACTIONS.items() if "n't" in v}
    for s in sentences:
        toks = re.findall(r"[A-Za-z']+", s.lower())
        signals = sum(
            1 for t in toks
            if t in NEG_FOLLOW or t.endswith("n't") or t == "not"
            or t in neg_typos
        )
        has_follow = any(t in NEG_FOLLOW for t in toks)
        if signals >= 2 and has_follow:
            issues.append({
                "type": "double negative",
                "bad": s,
                "suggestion": "rephrase with a single negative",
            })

    # 5. sentence not starting with a capital letter
    for s in sentences:
        if s and s[0].isalpha() and not s[0].isupper():
            issues.append({
                "type": "capitalization",
                "bad": s[:40] + ("..." if len(s) > 40 else ""),
                "suggestion": f"capitalize '{s[0]}'",
            })

    # 6. repeated terminal punctuation / spacing before punctuation
    for m in re.finditer(r"\s+([,.!?;:])", text):
        issues.append({
            "type": "spacing",
            "bad": repr(m.group(0)),
            "suggestion": f"no space before '{m.group(1)}'",
        })

    return issues


def check_spacy_optional(text):
    """Extra POS-based checks. Returns [] if spaCy or its model is absent."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return []  # not installed -> silently skip

    issues = []
    doc = nlp(text)
    for tok in doc:
        # singular subject pronoun + base-form verb: "he go", "she dont"
        if tok.dep_ == "nsubj" and tok.lower_ in {"he", "she", "it"}:
            head = tok.head
            if head.pos_ == "VERB" and head.tag_ == "VB":
                issues.append({
                    "type": "subject-verb (spaCy)",
                    "bad": f"{tok.text} {head.text}",
                    "suggestion": f"{tok.text} {head.lemma_}s",
                })
    return issues


def check_grammar(text, sentences):
    issues = []
    issues += check_rules(text, sentences)
    issues += check_spacy_optional(text)
    issues += check_spelling(text)
    return issues


# ------------------------------------------------------------------ redundancy

def find_pleonasms(text):
    found, low = [], text.lower()
    for pattern, better in PLEONASMS.items():
        for m in re.finditer(pattern, low):
            found.append((m.group(), better))
    return found


def find_repeated_words(text, top_n=10):
    stop = {
        "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at",
        "for", "with", "is", "are", "was", "were", "be", "been", "it", "its",
        "this", "that", "these", "those", "as", "by", "from", "we", "you",
        "they", "he", "she", "i", "our", "their", "his", "her", "will", "not",
    }
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    counts = Counter(w for w in words if w not in stop)
    return [(w, c) for w, c in counts.most_common(top_n) if c > 2]


def find_redundant_sentences(sentences, threshold=0.45):
    if len(sentences) < 2:
        return []
    X = TfidfVectorizer().fit_transform(sentences)
    sim = cosine_similarity(X)
    pairs = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if sim[i][j] >= threshold:
                pairs.append((sim[i][j], sentences[i], sentences[j]))
    return sorted(pairs, reverse=True)


# ------------------------------------------------------------------------ main

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_text_pure.py <file.pdf>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Reading: {path}\n")
    text = extract_text(path)
    if not text:
        print("No extractable text. Scanned PDF? Those need OCR.")
        sys.exit(1)

    sentences = split_sentences(text)
    print(f"Extracted {len(text)} chars, {len(sentences)} sentences.\n")

    print("=" * 60)
    print("GRAMMAR / SPELLING ISSUES")
    print("=" * 60)
    grammar = check_grammar(text, sentences)
    if not grammar:
        print("None found.")
    for g in grammar:
        print(f"  [{g['type']}] '{g['bad']}'  ->  {g['suggestion']}")

    print("\n" + "=" * 60)
    print("REDUNDANT PHRASES")
    print("=" * 60)
    pleos = find_pleonasms(text)
    print("None found." if not pleos else "")
    for phrase, better in pleos:
        print(f"  '{phrase}'  ->  '{better}'")

    print("\n" + "=" * 60)
    print("OVERUSED WORDS")
    print("=" * 60)
    repeats = find_repeated_words(text)
    print("None found." if not repeats else "")
    for word, count in repeats:
        print(f"  '{word}' used {count} times")

    print("\n" + "=" * 60)
    print("NEAR-DUPLICATE SENTENCES")
    print("=" * 60)
    dupes = find_redundant_sentences(sentences)
    print("None found." if not dupes else "")
    for score, s1, s2 in dupes:
        print(f"\n  Similarity {score:.2f}:")
        print(f"    - {s1}")
        print(f"    - {s2}")


if __name__ == "__main__":
    main()
