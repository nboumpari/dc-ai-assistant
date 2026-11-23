"""Chunking strategies on a small synthetic document: offsets must be exact, tables never cut mid-row."""

import pytest

from scripts.rules_corpus import chunk as ch

SECTION_TEXT = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 12

DOC = "\n\n".join(
    [
        "# 1 Introduction",
        SECTION_TEXT,
        "## 1.1 Scope",
        SECTION_TEXT * 3,
        "| AI | Meaning |\n|---|---|\n"
        + "\n".join(f"| ({i:02d}) | description of identifier {i} |" for i in range(60)),
        "## Article 6",
        "The daily driving time shall not exceed nine hours. " * 5,
    ]
)


def _sections(text: str) -> list[dict]:
    from scripts.rules_corpus.extract import find_sections

    return find_sections(text)


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "structure"])
def test_spans_cover_text_and_respect_limits(strategy):
    spans = ch.structure_spans(DOC, _sections(DOC)) if strategy == "structure" else ch.STRATEGIES[strategy](DOC)
    assert spans, "no chunks"
    assert spans[0][0] == 0
    assert max(end for _, end in spans) == len(DOC)
    limit = ch.STRUCTURE_MAX if strategy == "structure" else ch.CHUNK_SIZE
    assert all(end - start <= limit + 200 for start, end in spans)


@pytest.mark.parametrize("strategy", ["fixed", "recursive"])
def test_no_cut_inside_a_table_row(strategy):
    for start, end in ch.STRATEGIES[strategy](DOC):
        for cut in (start, end):
            if 0 < cut < len(DOC):
                line_start = DOC.rfind("\n", 0, cut) + 1
                line_end = DOC.find("\n", cut)
                line = DOC[line_start : line_end if line_end != -1 else len(DOC)]
                if line.lstrip().startswith("|"):
                    assert cut in (line_start, line_end), f"cut at {cut} splits the row {line!r}"


def test_structure_keeps_articles_whole():
    spans = ch.structure_spans(DOC, _sections(DOC))
    article_start = DOC.index("## Article 6")
    assert any(start <= article_start and end >= len(DOC) for start, end in spans)


def test_build_chunks_text_matches_offsets(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(ch, "TEXT_DIR", tmp_path)
    (tmp_path / "demo.md").write_text(DOC, encoding="utf-8")
    meta = {
        "title": "Demo document",
        "sections": _sections(DOC),
        "pages": [{"page": 1, "start": 0, "end": len(DOC) // 2}, {"page": 2, "start": len(DOC) // 2, "end": len(DOC)}],
    }
    (tmp_path / "demo.json").write_text(json.dumps(meta), encoding="utf-8")
    for strategy in ch.STRATEGIES:
        for c in ch.build_chunks("demo", strategy):
            assert c.text == DOC[c.char_start : c.char_end]
            assert c.text == c.text.strip()
            assert c.pages and set(c.pages) <= {1, 2}
            assert c.context.startswith("Demo document")
