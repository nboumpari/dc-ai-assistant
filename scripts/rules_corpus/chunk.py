"""Split the extracted documents into chunks with three strategies.

Run from the project root, after extract:

    uv run python -m scripts.rules_corpus.chunk

Writes data/documents/chunks/<strategy>.jsonl (gitignored), one chunk per line:

    chunk_id, strategy, document, section_id, section_title,
    char_start, char_end, pages, text, context

`text` is exactly the document text between char_start and char_end, so a
golden-set evidence span can be matched against any strategy's chunks by
overlap. `context` ("<document title> > <section title>") is the same kind of
header for every strategy; retrieval may prepend it when embedding.

Strategies:
    fixed      ~2000 characters (~500 tokens) with 300 characters overlap,
               cut at whitespace
    recursive  paragraphs, then lines, then sentences, then words, merged up
               to ~2000 characters, 200 characters overlap
    structure  one chunk per section or article; long sections split at
               paragraph boundaries, very short ones merged with the next
Markdown table rows are never cut in the middle.
"""

import bisect
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT_DIR = ROOT / "data" / "documents" / "text"
CHUNK_DIR = ROOT / "data" / "documents" / "chunks"

CHUNK_SIZE = 2000
FIXED_OVERLAP = 300
RECURSIVE_OVERLAP = 200
STRUCTURE_MAX = 4000
STRUCTURE_MIN = 300

Span = tuple[int, int]


@dataclass
class Chunk:
    chunk_id: str
    strategy: str
    document: str
    section_id: str | None
    section_title: str | None
    char_start: int
    char_end: int
    pages: list[int]
    text: str
    context: str


# ---------------------------------------------------------------- helpers


def _trim(text: str, span: Span) -> Span:
    """Shrink a span so it does not start or end with whitespace."""
    start, end = span
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _safe_cut(text: str, position: int, lower: int) -> int:
    """A cut point at or before `position` (and after `lower`) that is whitespace and not inside a table row."""
    for cut in range(position, lower, -1):
        if not text[cut - 1].isspace():
            continue
        line_start = text.rfind("\n", 0, cut) + 1
        line_end = text.find("\n", cut)
        line = text[line_start : line_end if line_end != -1 else len(text)]
        if line.lstrip().startswith("|") and cut not in (line_start, line_end):
            continue  # inside a table row
        return cut
    return position


def _pieces(text: str, start: int, end: int, separators: list[str]) -> list[Span]:
    """Split [start, end) at the first separator that occurs, recursing with finer separators."""
    if end - start <= CHUNK_SIZE or not separators:
        return [(start, end)]
    sep, finer = separators[0], separators[1:]
    pattern = re.compile(sep)
    cuts = [m.end() for m in pattern.finditer(text, start, end)]
    if not cuts:
        return _pieces(text, start, end, finer)
    spans, last = [], start
    for cut in cuts + [end]:
        if cut > last:
            spans.extend(_pieces(text, last, cut, finer))
            last = cut
    return spans


# ---------------------------------------------------------------- strategies


def fixed_spans(text: str) -> list[Span]:
    spans, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            end = _safe_cut(text, end, start + CHUNK_SIZE // 2)
        spans.append((start, end))
        if end >= len(text):
            break
        start = _safe_cut(text, max(end - FIXED_OVERLAP, start + 1), start)
    return spans


def recursive_spans(text: str) -> list[Span]:
    separators = [r"\n\s*\n", r"\n", r"(?<=[.;:])\s+", r"\s+"]  # a line break between table rows is a safe cut
    pieces = _pieces(text, 0, len(text), separators)
    spans, current = [], None
    for start, end in pieces:
        if current is None:
            current = [start, end]
        elif end - current[0] <= CHUNK_SIZE:
            current[1] = end
        else:
            spans.append((current[0], current[1]))
            overlap_start = _safe_cut(text, max(current[1] - RECURSIVE_OVERLAP, current[0] + 1), current[0])
            current = [min(overlap_start, start), end]
    if current:
        spans.append((current[0], current[1]))
    return spans


def structure_spans(text: str, sections: list[dict]) -> list[Span]:
    """One span per leaf section; long ones split at paragraph boundaries, short ones merged forward."""
    if not sections:
        return recursive_spans(text)
    bounds = [(0, sections[0]["start"])] if sections[0]["start"] > 0 else []
    bounds += [(s["start"], s["end"]) for s in sections]
    merged: list[list[int]] = []
    for start, end in bounds:
        if merged and merged[-1][1] - merged[-1][0] < STRUCTURE_MIN:
            merged[-1][1] = end  # too short: absorb the next section
        else:
            merged.append([start, end])
    spans: list[Span] = []
    for start, end in merged:
        if end - start <= STRUCTURE_MAX:
            spans.append((start, end))
            continue
        current = start
        for p_start, p_end in _pieces(text, start, end, [r"\n\s*\n", r"\n", r"(?<=[.;:])\s+"]):
            if p_end - current > STRUCTURE_MAX and p_start > current:
                spans.append((current, p_start))
                current = p_start
        spans.append((current, end))
    return spans


STRATEGIES = {"fixed": fixed_spans, "recursive": recursive_spans, "structure": structure_spans}


# ---------------------------------------------------------------- building


def build_chunks(doc_id: str, strategy: str) -> list[Chunk]:
    text = (TEXT_DIR / f"{doc_id}.md").read_text(encoding="utf-8")
    meta = json.loads((TEXT_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))
    sections = meta["sections"]
    section_starts = [s["start"] for s in sections]
    page_starts = [p["start"] for p in meta["pages"]]

    raw = structure_spans(text, sections) if strategy == "structure" else STRATEGIES[strategy](text)
    chunks = []
    for start, end in raw:
        start, end = _trim(text, (start, end))
        if end <= start:
            continue
        i = bisect.bisect_right(section_starts, start) - 1
        section = sections[i] if i >= 0 else None
        first = max(bisect.bisect_right(page_starts, start) - 1, 0)
        last = max(bisect.bisect_right(page_starts, end - 1) - 1, 0)
        pages = sorted({p["page"] for p in meta["pages"][first : last + 1] if p["page"]})
        title = section["title"] if section else None
        chunks.append(
            Chunk(
                chunk_id=f"{strategy}:{doc_id}:{len(chunks):04d}",
                strategy=strategy,
                document=doc_id,
                section_id=section["id"] if section else None,
                section_title=title,
                char_start=start,
                char_end=end,
                pages=pages,
                text=text[start:end],
                context=f"{meta['title']} > {title}" if title else meta["title"],
            )
        )
    return chunks


def main() -> int:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    documents = sorted(p.stem for p in TEXT_DIR.glob("*.md"))
    if not documents:
        print("No extracted documents; run extract first.", file=sys.stderr)
        return 1
    for strategy in STRATEGIES:
        chunks = [c for doc_id in documents for c in build_chunks(doc_id, strategy)]
        with (CHUNK_DIR / f"{strategy}.jsonl").open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        sizes = sorted(len(c.text) for c in chunks)
        print(
            f"  {strategy:<10} {len(chunks):>5} chunks   median {sizes[len(sizes) // 2]:>5} chars   "
            f"max {sizes[-1]:>5}   documents {len(documents)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
