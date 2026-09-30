"""
preprocess.py
-------------
PDF -> extracted text -> cleaned text -> chunks -> train/test split (BY DOCUMENT) -> saved files.

Expected layout (this file lives in src/):
    project/
      data/                 <- input PDFs:  ../data/<file-name>.pdf   (relative to src/)
      src/preprocess.py
      outputs/              <- not used here

Outputs (written to ../data/processed/):
    extracted/<doc_id>.txt  cached raw text (so PDFs are parsed only once)
    chunks_all.jsonl        every chunk, with a "split" column
    train.jsonl             chunks from training documents
    test.jsonl              chunks from held-out documents
    stats.json              small summary (docs, chunks, words per chunk, skipped files)

Each record:
    {"chunk_id", "doc_id", "chunk_idx", "split", "n_words", "text"}

Dependencies:  pip install pypdf   (and/or pymupdf, see EXTRACTOR)
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

try:
    import pymupdf  # PyMuPDF (pip install pymupdf)
except ImportError:
    pymupdf = None
try:
    from pypdf import PdfReader  # pip install pypdf
except ImportError:
    PdfReader = None

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / ".." / "data").resolve()
OUT_DIR = DATA_DIR / "processed"
EXTRACT_DIR = OUT_DIR / "extracted"

CHUNK_WORDS = 100        # target words per chunk (~130 tokens -> fits max_seq_length=128-192)
OVERLAP_SENTENCES = 1    # sentences repeated between consecutive chunks
MIN_CHUNK_WORDS = 20     # drop / merge chunks shorter than this
MAX_CHUNK_WORDS = 160    # hard cap; longer sentences are split by words
TEST_FRACTION = 0.4      # fraction of DOCUMENTS held out
SEED = 42
MIN_CHARS_PER_PAGE = 50  # below this average the PDF is treated as scanned
REMOVE_REFERENCES = True # cut "References"/"Bibliography" sections
EXTRACTOR = "pypdf"      # "pypdf" or "pymupdf"


# --------------------------------------------------------------------------- #
# 1. Extraction
# --------------------------------------------------------------------------- #
def extract_pdf(path: Path) -> list[str]:
    """Return a list with one string per page (reading order, multi-column aware)."""
    if EXTRACTOR == "pypdf":
        if PdfReader is None:
            raise ImportError("pip install pypdf")
        reader = PdfReader(str(path))
        return [(page.extract_text() or "") for page in reader.pages]

    if pymupdf is None:
        raise ImportError("pip install pymupdf")
    with pymupdf.open(path) as doc:
        return [page.get_text("text", sort=True) for page in doc]


def is_scanned(pages: list[str]) -> bool:
    if not pages:
        return True
    avg = sum(len(p.strip()) for p in pages) / len(pages)
    return avg < MIN_CHARS_PER_PAGE


# --------------------------------------------------------------------------- #
# 2. Cleaning
# --------------------------------------------------------------------------- #
def _normalize_line(line: str) -> str:
    """Key used to detect repeated headers/footers: digits collapsed to '#'."""
    return re.sub(r"\d+", "#", line.strip().lower())


def strip_boilerplate(pages: list[str], edge_lines: int = 3, min_frac: float = 0.4) -> list[str]:
    """
    Remove lines that repeat across many pages at the top/bottom of a page
    (running headers, footers, page numbers) plus bare page-number lines.
    """
    if len(pages) < 4:  # too few pages to detect repetition reliably
        edge_repeats: set[str] = set()
    else:
        counter: Counter[str] = Counter()
        for p in pages:
            lines = [l for l in p.splitlines() if l.strip()]
            edges = lines[:edge_lines] + lines[-edge_lines:]
            for key in {_normalize_line(l) for l in edges}:
                counter[key] += 1
        edge_repeats = {k for k, c in counter.items() if c / len(pages) >= min_frac and k}

    page_num = re.compile(r"^\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*$", re.I)
    cleaned = []
    for p in pages:
        lines = p.splitlines()
        nonempty_idx = [i for i, l in enumerate(lines) if l.strip()]
        edge_idx = set(nonempty_idx[:edge_lines] + nonempty_idx[-edge_lines:])
        kept = []
        for i, l in enumerate(lines):
            if page_num.match(l):
                continue
            if i in edge_idx and _normalize_line(l) in edge_repeats:
                continue
            kept.append(l)
        cleaned.append("\n".join(kept))
    return cleaned


def cut_references(text: str) -> str:
    """Drop everything after a 'References' / 'Bibliography' heading in the last 40% of the doc."""
    m = None
    for m in re.finditer(r"^\s*(references|bibliography|works cited)\s*$", text, re.I | re.M):
        pass  # keep the last match
    if m and m.start() > 0.6 * len(text):
        return text[: m.start()]
    return text


def clean(text: str) -> str:
    """Clean a full document's text. Paragraph boundaries (blank lines) are preserved."""
    text = unicodedata.normalize("NFKC", text)           # ligatures: ﬁ -> fi
    text = text.replace("\u00ad", "")                    # soft hyphens
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)  # control chars
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)         # de-hyphenate line breaks
    text = text.replace("\r", "")

    if REMOVE_REFERENCES:
        text = cut_references(text)

    # Join hard-wrapped lines inside a paragraph, keep blank lines as paragraph breaks
    paragraphs = re.split(r"\n\s*\n", text)
    out = []
    for para in paragraphs:
        para = re.sub(r"\s*\n\s*", " ", para)            # single newline -> space
        para = re.sub(r"[ \t]+", " ", para).strip()
        if len(para.split()) >= 5:                       # drop stray fragments (labels, axis ticks)
            out.append(para)
    return "\n\n".join(out)


# --------------------------------------------------------------------------- #
# 3. Chunking
# --------------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“(\[])')


def _split_sentences(text: str) -> list[str]:
    sents = []
    for para in text.split("\n\n"):
        sents.extend(s.strip() for s in _SENT_SPLIT.split(para) if s.strip())
    return sents


def _hard_split(sentence: str, max_words: int) -> list[str]:
    words = sentence.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def chunk(text: str,
          target_words: int = CHUNK_WORDS,
          overlap_sentences: int = OVERLAP_SENTENCES,
          min_words: int = MIN_CHUNK_WORDS,
          max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Pack consecutive sentences into ~target_words chunks with sentence overlap."""
    sents = []
    for s in _split_sentences(text):
        sents.extend(_hard_split(s, max_words) if len(s.split()) > max_words else [s])

    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_words = 0
    for s in sents:
        w = len(s.split())
        if cur and cur_words + w > target_words:
            chunks.append(cur)
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            cur_words = sum(len(x.split()) for x in cur)
        cur.append(s)
        cur_words += w
    if cur:
        chunks.append(cur)

    texts = [" ".join(c) for c in chunks]

    # merge a short trailing chunk into the previous one, drop other short chunks
    if len(texts) >= 2 and len(texts[-1].split()) < min_words:
        texts[-2] = texts[-2] + " " + texts[-1]
        texts.pop()
    return [t for t in texts if len(t.split()) >= min_words]


# --------------------------------------------------------------------------- #
# 4. Split (by document, never by chunk -> avoids leakage)
# --------------------------------------------------------------------------- #
def split_docs(doc_ids: list[str], test_fraction: float = TEST_FRACTION, seed: int = SEED) -> dict[str, str]:
    ids = sorted(doc_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    if len(ids) < 2:
        return {d: "train" for d in ids}
    n_test = max(1, round(len(ids) * test_fraction))
    test = set(ids[:n_test])
    return {d: ("test" if d in test else "train") for d in ids}


# --------------------------------------------------------------------------- #
# 5. Save helpers
# --------------------------------------------------------------------------- #
def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_chunks(split: str = "all", processed_dir: Path = OUT_DIR) -> list[dict]:
    """Convenience loader for later stages: split in {'all', 'train', 'test'}."""
    name = "chunks_all.jsonl" if split == "all" else f"{split}.jsonl"
    with open(processed_dir / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --------------------------------------------------------------------------- #
# 6. Pipeline
# --------------------------------------------------------------------------- #
def run_preprocess(data_dir: Path = DATA_DIR, out_dir: Path = OUT_DIR,
                   use_cache: bool = True) -> dict:
    extract_dir = out_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")

    doc_chunks: dict[str, list[str]] = {}
    skipped: list[dict] = []

    for pdf in pdfs:
        doc_id = pdf.stem
        cache = extract_dir / f"{doc_id}.txt"

        if use_cache and cache.exists():
            text = cache.read_text(encoding="utf-8")
        else:
            try:
                pages = extract_pdf(pdf)
            except Exception as e:  # corrupt / encrypted file
                skipped.append({"doc_id": doc_id, "reason": f"read error: {e}"})
                continue
            if is_scanned(pages):
                skipped.append({"doc_id": doc_id, "reason": "scanned/empty (needs OCR)"})
                continue
            text = clean("\n".join(strip_boilerplate(pages)))
            cache.write_text(text, encoding="utf-8")

        chunks = chunk(text)
        if not chunks:
            skipped.append({"doc_id": doc_id, "reason": "no usable chunks after cleaning"})
            continue
        doc_chunks[doc_id] = chunks

    if not doc_chunks:
        raise RuntimeError(f"All PDFs were skipped: {skipped}")

    split_map = split_docs(list(doc_chunks))
    records = []
    for doc_id, chunks in doc_chunks.items():
        for i, t in enumerate(chunks):
            records.append({
                "chunk_id": f"{doc_id}__{i:04d}",
                "doc_id": doc_id,
                "chunk_idx": i,
                "split": split_map[doc_id],
                "n_words": len(t.split()),
                "text": t,
            })

    train = [r for r in records if r["split"] == "train"]
    test = [r for r in records if r["split"] == "test"]
    _write_jsonl(out_dir / "chunks_all.jsonl", records)
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "test.jsonl", test)

    words = [r["n_words"] for r in records]
    stats = {
        "n_docs": len(doc_chunks),
        "n_train_docs": sum(v == "train" for v in split_map.values()),
        "n_test_docs": sum(v == "test" for v in split_map.values()),
        "test_docs": sorted(d for d, s in split_map.items() if s == "test"),
        "n_chunks": len(records),
        "n_train_chunks": len(train),
        "n_test_chunks": len(test),
        "words_per_chunk": {"min": min(words), "mean": round(sum(words) / len(words), 1), "max": max(words)},
        "chunks_per_doc": {d: len(c) for d, c in doc_chunks.items()},
        "skipped": skipped,
        "config": {"chunk_words": CHUNK_WORDS, "overlap_sentences": OVERLAP_SENTENCES,
                   "min_chunk_words": MIN_CHUNK_WORDS, "test_fraction": TEST_FRACTION, "seed": SEED},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


if __name__ == "__main__":
    s = run_preprocess()
    print(json.dumps({k: v for k, v in s.items() if k != "chunks_per_doc"}, indent=2))
    print("chunks per doc:", s["chunks_per_doc"])
