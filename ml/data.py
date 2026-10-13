"""Download and clean the Kazakh classics corpus from Wikisource.

The Kazakh section of Wikisource lives on the multilingual wiki (wikisource.org),
not on kk.wikisource.org. We pull the raw wikitext of every page in a few author
categories through the MediaWiki API (50 pages per request, throttled), strip the
markup, normalise the text and keep a compact character vocabulary.

Outputs (all under ml/data/, gitignored):
  raw/pages.json   cached wikitext, {title: wikitext}
  corpus.txt       cleaned text, documents separated by a blank line
  meta.json        sources, per-author character counts, vocabulary

Usage:  python ml/data.py            (uses cache if present)
        python ml/data.py --refresh  (re-download)
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw" / "pages.json"
CORPUS = DATA / "corpus.txt"
META = DATA / "meta.json"

API = "https://wikisource.org/w/api.php"
UA = "GlassboxCorpusFetcher/0.1 (educational portfolio project; character-level GPT)"

# Author categories on wikisource.org (Kazakh section). All authors died before 1954,
# so their works are in the public domain (Kazakhstan: life + 70 years).
SOURCES = [
    # (category, author label, died)
    ("Абай Құнанбайұлы", "Abai Kunanbayuly", 1904),
    ("Ыбырай Алтынсарин", "Ybyrai Altynsarin", 1889),
    ("Ахмет Байтұрсынұлы", "Akhmet Baitursynuly", 1937),
    ("Мағжан Жұмабаев", "Magzhan Zhumabayev", 1938),
]
# Individual pages outside those categories.
EXTRA_PAGES = [
    ("Шортанбайдың өлер алдында қатын-бала, ел-жұртына арыздасып айтқан насихат сөздері",
     "Shortanbai Kanaiuly", 1881),
    ("Зар заман", "Shortanbai Kanaiuly", 1881),
]

MIN_COUNT = 5  # characters rarer than this are dropped from the corpus

# Author index pages (lists of titles) and a page that was vandalised on Wikisource
# (emoji / keyboard-mash inserted into an Abai poem as of 2026).
EXCLUDE = {c for c, _, _ in SOURCES} | {"Сағаттың шықылдағы емес ермек"}
# Stray non-literary lines found in the dump (vandalism).
JUNK_LINES = [re.compile(r"^Остался \d+ день$")]

session = requests.Session()
session.headers["User-Agent"] = UA


def api(**params) -> dict:
    params |= {"format": "json", "formatversion": 2}
    for attempt in range(10):
        time.sleep(2.0)
        r = session.get(API, params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            wait = int(r.headers.get("retry-after", 5)) + 2 ** attempt
            print(f"  HTTP {r.status_code}, backing off {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Wikisource API kept rate-limiting us")


def category_titles(cat: str, seen: set[str] | None = None) -> list[str]:
    """Main-namespace pages in a category, recursing into subcategories."""
    seen = seen if seen is not None else set()
    titles: list[str] = []
    cont: dict = {}
    while True:
        d = api(action="query", list="categorymembers", cmtitle=f"Category:{cat}",
                cmlimit=500, cmprop="title|ns", **cont)
        for m in d["query"]["categorymembers"]:
            if m["ns"] == 14:
                sub = m["title"].split(":", 1)[1]
                if sub not in seen:
                    seen.add(sub)
                    titles += category_titles(sub, seen)
            elif m["ns"] == 0:
                titles.append(m["title"])
        if "continue" not in d:
            return titles
        cont = d["continue"]


def fetch_wikitext(titles: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        d = api(action="query", prop="revisions", rvprop="content", rvslots="main",
                titles="|".join(batch), redirects=1)
        for p in d["query"]["pages"]:
            revs = p.get("revisions")
            if revs:
                out[p["title"]] = revs[0]["slots"]["main"]["content"]
        print(f"  fetched {len(out)}/{len(titles)}", file=sys.stderr)
    return out


def download() -> dict:
    """Returns {"pages": {title: wikitext}, "author_of": {title: author}}."""
    author_of: dict[str, str] = {}
    for cat, author, _ in SOURCES:
        ts = sorted(set(category_titles(cat)))
        print(f"{author}: {len(ts)} pages", file=sys.stderr)
        for t in ts:
            author_of.setdefault(t, author)
    for t, author, _ in EXTRA_PAGES:
        author_of.setdefault(t, author)
    pages = fetch_wikitext(sorted(author_of))
    return {"pages": pages, "author_of": author_of}


# --------------------------------------------------------------------------- cleaning

CYR = "А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі"
# Latin letters that are visually identical to Cyrillic ones and often sneak into
# transcriptions (e.g. "өзi" typed with a Latin i).
HOMOGLYPHS = str.maketrans({
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "i": "і",
    "A": "А", "E": "Е", "O": "О", "P": "Р", "C": "С", "X": "Х", "H": "Н", "K": "К",
    "M": "М", "T": "Т", "B": "В", "I": "І",
})
PUNCT_MAP = str.maketrans({
    "–": "—", "‒": "—", "―": "—", "−": "-", "‐": "-", "‑": "-",
    "“": "«", "„": "«", "”": "»", "‟": "»", "’": "'", "‘": "'", "`": "'", "´": "'",
    " ": " ", " ": " ", " ": " ", "\t": " ",
})


def strip_templates(s: str) -> str:
    """Remove {{...}} templates, which may nest."""
    out, depth, i = [], 0, 0
    while i < len(s):
        if s.startswith("{{", i):
            depth += 1
            i += 2
        elif s.startswith("}}", i) and depth:
            depth -= 1
            i += 2
        else:
            if depth == 0:
                out.append(s[i])
            i += 1
    return "".join(out)


def is_index_page(wikitext: str) -> bool:
    """Pages that are mostly bullet lists are author indexes, not works."""
    lines = [l for l in wikitext.splitlines() if l.strip()]
    if not lines:
        return True
    listy = sum(1 for l in lines if re.match(r"\s*[*#]\s*\[\[", l))
    return listy / len(lines) > 0.3


def clean_wikitext(w: str) -> str:
    s = w.replace("\r", "")
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.S)
    s = re.sub(r"<references\s*/?>", "", s)
    # Drop everything from a notes / footnotes / source section onwards.
    s = re.split(r"\n=+\s*(Ескертпелер|Ескертпе|Сілтемелер|Дереккөздер|Түсініктеме\w*)\s*=+", s)[0]
    s = re.sub(r"\{\|.*?\|\}", "", s, flags=re.S)  # tables
    s = strip_templates(s)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</?[a-zA-Z][^>]*>", "", s)  # remaining tags (<poem>, <center>, ...)
    s = re.sub(r"\[\[(?:File|Image|Файл|Сурет|Category|Санат|w|wikipedia|:w|:wikipedia)?:[^\]]*\]\]", "", s, flags=re.I)
    s = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", s)
    s = re.sub(r"\[?https?://\S+\]?", "", s)
    s = html.unescape(s)
    s = unicodedata.normalize("NFC", s)
    s = s.translate(PUNCT_MAP)
    s = s.replace("…", "...")

    lines = []
    for line in s.split("\n"):
        t = line.strip()
        if re.fullmatch(r"=+.*=+", t):          # section headers
            continue
        if re.fullmatch(r"''[^']+''", t):        # italic-only line: author byline / editorial note
            continue
        if t.startswith("__") or t.startswith("[["):
            continue
        if any(j.match(t) for j in JUNK_LINES):
            continue
        t = re.sub(r"^[*#:;>]+\s*", "", t)       # list / indent / quote markers
        if re.fullmatch(r"[*\s]*", t):          # "* * *" stanza separators
            lines.append("")
            continue
        t = t.replace("'''", "").replace("''", "")
        # Fix Latin homoglyphs inside Cyrillic words.
        t = re.sub(rf"(?<=[{CYR}])[A-Za-z]+|[A-Za-z]+(?=[{CYR}])", lambda m: m.group(0).translate(HOMOGLYPHS), t)
        if re.search(r"[A-Za-z]", t):
            latin = len(re.findall(r"[A-Za-z]", t))
            cyr = len(re.findall(rf"[{CYR}]", t))
            if latin > cyr:                       # mostly-Latin line: junk
                continue
            t = re.sub(r"\b[A-Za-z][A-Za-z.\-]*\b", "", t)
        # Digits are rare (footnote markers glued to words like "бодам1", a few years);
        # drop them to keep the vocabulary purely textual.
        t = re.sub(r"\d+", "", t)
        t = re.sub(r" {2,}", " ", t).strip()
        lines.append(t)
    s = "\n".join(lines)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def build(refresh: bool = False) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if RAW.exists() and not refresh:
        raw = json.loads(RAW.read_text())
    else:
        raw = download()
        RAW.parent.mkdir(parents=True, exist_ok=True)
        RAW.write_text(json.dumps(raw, ensure_ascii=False, indent=0))

    docs: list[tuple[str, str, str]] = []  # (author, title, text)
    for title, wt in raw["pages"].items():
        author = raw["author_of"].get(title, "?")
        if title in EXCLUDE or wt.lstrip().lower().startswith("#redirect"):
            continue
        if is_index_page(wt):
            continue
        text = clean_wikitext(wt)
        if len(text) < 80:
            continue
        docs.append((author, title, text))

    # Shuffle documents (fixed seed) so the last-10% validation split mixes authors.
    docs.sort(key=lambda d: d[1])
    random.Random(1904).shuffle(docs)
    text = "\n\n".join(d[2] for d in docs) + "\n"

    counts = Counter(text)
    rare = {c for c, n in counts.items() if n < MIN_COUNT}
    if rare:
        text = "".join(c for c in text if c not in rare)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r" +\n", "\n", text)
    vocab = sorted(set(text))

    per_author = Counter()
    for a, _, t in docs:
        per_author[a] += len(t)
    CORPUS.write_text(text)
    meta = {
        "name": "Kazakh classics: Abai Kunanbayuly and contemporaries",
        "source_url": "https://wikisource.org/wiki/Category:Kazakh",
        "source_urls": [f"https://wikisource.org/wiki/Category:{c.replace(' ', '_')}" for c, _, _ in SOURCES]
        + [f"https://wikisource.org/wiki/{t.replace(' ', '_')}" for t, _, _ in EXTRA_PAGES],
        "license": "Public domain: all authors died 1881-1938 (Kazakhstan copyright term is life + 70 years). "
                   "Transcriptions from Wikisource; editorial footnotes removed.",
        "authors": {a: {"died": d} for _, a, d in SOURCES + EXTRA_PAGES},
        "documents": len(docs),
        "chars_per_author": dict(per_author.most_common()),
        "chars": len(text),
        "dropped_rare_chars": "".join(sorted(rare)),
        "vocab": vocab,
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"{len(docs)} documents, {len(text):,} chars, vocab {len(vocab)}")
    print("per author:", dict(per_author.most_common()))
    print("dropped rare:", repr("".join(sorted(rare))))
    print("vocab:", repr("".join(vocab)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    build(ap.parse_args().refresh)
