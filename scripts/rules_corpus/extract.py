"""Turn the downloaded documents into clean text with page and section positions.

Run from the project root, after download:

    uv run python -m scripts.rules_corpus.extract

For each document this writes, to data/documents/text/ (gitignored):

    <id>.md    one Markdown text: headings, paragraphs and tables
    <id>.json  where every page and every section starts and ends in that text
               (character offsets), plus the document's details

Chunking and the golden set both refer to these offsets, so a passage can be
located in the same way whatever chunking strategy is used.

PDFs are converted with pymupdf4llm. Documents with a `sections` list keep
only those sections, found through the PDF's table of contents. Headers and
footers repeated on most pages are removed. Lines such as "Article 6" are
treated as headings in legal texts.
"""

import html
import json
import re
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import pymupdf
import pymupdf4llm
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("corpus.yaml")
RAW_DIR = ROOT / "data" / "documents" / "raw"
TEXT_DIR = ROOT / "data" / "documents" / "text"

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
ARTICLE = re.compile(r"^[*_\s]*(?:Article|ARTICLE)\s+([0-9IlOJ]{1,3}[a-z]?)[*_\s]*$")
OCR_DIGITS = str.maketrans({"I": "1", "l": "1", "J": "1", "O": "0"})
SECTION_NUMBER = re.compile(r"^\**\s*(\d+(?:\.\d+)*)\.?\s+(.*)$")


# ---------------------------------------------------------------- PDF pages


def selected_pages(pdf_path: Path, sections: list[str]) -> list[int]:
    """0-based page numbers covering the given section numbers, using the PDF's table of contents."""
    toc = pymupdf.open(pdf_path).get_toc()
    entries = [(lvl, title.split(" ")[0].rstrip("."), page) for lvl, title, page in toc]
    pages: set[int] = set()
    for wanted in sections:
        for i, (lvl, number, start) in enumerate(entries):
            if number != wanted:
                continue
            end = next((p for lv, _, p in entries[i + 1 :] if lv <= lvl), start)
            pages.update(range(start - 1, max(start, end)))  # include the page where the next section starts
            break
        else:
            raise ValueError(f"{pdf_path.name}: section {wanted} not found in the table of contents")
    return sorted(pages)


def pdf_pages(pdf_path: Path, pages: list[int] | None) -> list[tuple[int, str]]:
    """[(1-based page number, markdown)] for the requested pages."""
    chunks = pymupdf4llm.to_markdown(str(pdf_path), pages=pages, page_chunks=True, show_progress=False)
    return [(c["metadata"]["page_number"], c["text"]) for c in chunks]


def remove_repeated_lines(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop header/footer lines that appear on more than half of the pages (page numbers ignored)."""

    def key(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().strip("*_ ").lower())

    def is_heading(line: str) -> bool:
        stripped = line.strip()
        return bool(HEADING.match(stripped) or ARTICLE.match(stripped))

    counts = Counter()
    for _, text in pages:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        counts.update({key(ln) for ln in lines[:4] + lines[-4:] if not is_heading(ln)})
    repeated = {k for k, n in counts.items() if n > len(pages) / 2 and len(pages) > 3}
    cleaned = []
    for number, text in pages:
        kept = [ln for ln in text.splitlines() if is_heading(ln) or not (ln.strip() and key(ln) in repeated)]
        cleaned.append((number, "\n".join(kept)))
    return cleaned


# ---------------------------------------------------------------- XHTML


class _XhtmlToMarkdown(HTMLParser):
    """Minimal XHTML-to-Markdown: block elements become paragraphs, table cells become rows."""

    BLOCKS = {"p", "div", "h1", "h2", "h3", "h4", "li", "tr", "br"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self.skip += 1
        if tag in self.BLOCKS:
            self.parts.append("\n\n")
        if tag in ("td", "th"):
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def xhtml_text(path: Path) -> str:
    parser = _XhtmlToMarkdown()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    text = html.unescape("".join(parser.parts)).replace("\xa0", " ")
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paragraphs if p)


# ---------------------------------------------------------------- excerpts (PDFs without a table of contents)

DEFINITION = re.compile(r'^[-\s]*_?"([^"]+)"')


def _line_pattern(regex: str) -> re.Pattern:
    """A heading or list marker may come before the start/end text: '# ', '**', '- ', '_'."""
    return re.compile(r"(?m)^[#*_\-\s]*" + regex)


def _keep_definitions(text: str, terms: list[str]) -> str:
    """Keep definition paragraphs whose term contains one of the terms, with the notes that follow them."""
    kept, keep = [], False
    for paragraph in re.split(r"\n\s*\n", text):
        if m := DEFINITION.match(paragraph.strip()):
            term = m.group(1).lower().strip("_ ")
            keep = any(term.startswith(t) for t in terms)
        if keep:
            kept.append(paragraph)
    return "\n\n".join(kept)


def _keep_table_rows(text: str, pattern: str) -> str:
    """Keep a table's header rows and the rows whose first cell matches the pattern."""
    kept = []
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        first_cell = line.split("|")[1] if line.count("|") > 1 else ""
        if "---" in line or re.search(pattern, first_cell) or not re.search(r"\d{4}", first_cell):
            kept.append(line)
    return "\n".join(kept)


def selection_pages(selections: list[dict]) -> list[tuple[str, int, str]]:
    """Excerpts cut by page range and start/end text, each with its own heading; returns (file, page, text)."""
    wanted: dict[str, set[int]] = {}
    for s in selections:
        wanted.setdefault(s["file"], set()).update(range(s["pages"][0] - 1, s["pages"][1]))
    pages: dict[tuple[str, int], str] = {}
    for name, numbers in wanted.items():
        for number, text in remove_repeated_lines(pdf_pages(RAW_DIR / name, sorted(numbers))):
            pages[(name, number)] = text

    out: list[tuple[str, int, str]] = []
    for s in selections:
        numbers = range(s["pages"][0], s["pages"][1] + 1)
        segments, joined = [], ""
        for n in numbers:
            segments.append((n, len(joined)))
            joined += pages[(s["file"], n)] + "\n\n"
        start, end = 0, len(joined)
        if s.get("start"):
            m = _line_pattern(s["start"]).search(joined)
            if not m:
                raise ValueError(f"{s['heading']}: start {s['start']!r} not found")
            start = m.start()
        if s.get("end"):
            m = _line_pattern(s["end"]).search(joined, start + 1)
            end = m.start() if m else end
        out.append((s["file"], s["pages"][0], f"# {s['heading']}"))
        for i, (n, offset) in enumerate(segments):
            seg_end = segments[i + 1][1] if i + 1 < len(segments) else len(joined)
            piece = joined[max(offset, start) : min(seg_end, end)]
            if s.get("definitions"):
                piece = _keep_definitions(piece, s["definitions"])
            if s.get("table_rows"):
                piece = _keep_table_rows(piece, s["table_rows"])
            if piece.strip():
                out.append((s["file"], n, piece))
    return out


# ---------------------------------------------------------------- assembling


def tidy(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_sections(text: str) -> list[dict]:
    """Headings (Markdown '#' lines and 'Article N' lines) with their character ranges."""
    marks = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if (m := HEADING.match(stripped)) and len(m.group(2).strip("*_ ")) > 2:  # skip index letters
            title = m.group(2).strip("*_ ").strip()
            number = SECTION_NUMBER.match(title)
            level = len(m.group(1))
            if number:
                level = number.group(1).count(".") + 1
            marks.append({"id": number.group(1) if number else title, "title": title, "level": level, "start": offset})
        elif m := ARTICLE.match(stripped):
            number = m.group(1).translate(OCR_DIGITS)
            marks.append({"id": f"Article {number}", "title": f"Article {number}", "level": 2, "start": offset})
        offset += len(line)
    # "Article N" lines are matched in headings too: normalise them and merge the title heading that follows
    merged: list[dict] = []
    for mark in marks:
        if a := ARTICLE.match(mark["title"]):
            number = a.group(1).translate(OCR_DIGITS)
            mark.update(id=f"Article {number}", title=f"Article {number}", level=2)
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous["id"].startswith("Article ")
            and previous["title"] == previous["id"]
            and not mark["id"].startswith("Article ")
            and not SECTION_NUMBER.match(mark["title"])
            and text[previous["start"] : mark["start"]].count("\n") <= 3
        ):
            previous["title"] = f"{previous['id']} - {mark['title']}"
            continue
        merged.append(mark)
    marks = merged
    for i, mark in enumerate(marks):
        mark["end"] = marks[i + 1]["start"] if i + 1 < len(marks) else len(text)
    return marks


def build_document(doc: dict) -> tuple[str, dict]:
    files = [RAW_DIR / name for name in doc["files"]]
    page_texts: list[tuple[str, int, str]] = []  # (file, page number, text)
    if doc.get("selections"):
        page_texts = selection_pages(doc["selections"])
        files = []
    for path in files:
        if path.suffix == ".xhtml":
            page_texts.append((path.name, 0, xhtml_text(path)))
            continue
        pages = selected_pages(path, doc["sections"]) if doc.get("sections") else None
        for number, text in remove_repeated_lines(pdf_pages(path, pages)):
            page_texts.append((path.name, number, text))

    parts, page_index, offset = [], [], 0
    for name, number, text in page_texts:
        text = tidy(text)
        if not text:
            continue
        piece = text + "\n\n"
        page_index.append({"file": name, "page": number or None, "start": offset, "end": offset + len(text)})
        parts.append(piece)
        offset += len(piece)
    full = "".join(parts)
    meta = {k: doc[k] for k in ("id", "title", "theme", "publisher", "version", "url", "licence")}
    meta.update({"chars": len(full), "pages": page_index, "sections": find_sections(full)})
    return full, meta


def main() -> int:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    docs = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["documents"]
    for doc in docs:
        if only and doc["id"] not in only:
            continue
        missing = [n for n in doc["files"] if not (RAW_DIR / n).exists()]
        if missing:
            print(f"  skip {doc['id']}: missing {', '.join(missing)} (run download first)")
            continue
        text, meta = build_document(doc)
        (TEXT_DIR / f"{doc['id']}.md").write_text(text, encoding="utf-8")
        (TEXT_DIR / f"{doc['id']}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        pages = len([p for p in meta["pages"] if p["page"]]) or "-"
        print(f"  ok  {doc['id']:<28} pages {pages!s:>4}  sections {len(meta['sections']):>4}  chars {len(text):>9,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
