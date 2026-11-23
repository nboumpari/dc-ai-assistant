"""Retrieval metrics against the golden set's evidence spans.

Evidence is a list of {document, char_start, char_end}: the exact passages of
the extracted document text that answer the question. Chunks carry the same
coordinates, so a chunk is relevant if its span overlaps an evidence span in
the same document. This works for every chunking strategy alike.

  hit@k       1 if any of the top k chunks is relevant, else 0
  coverage@k  share of the evidence characters contained in the top k chunks
              (a question can need several passages; a chunk can hold only part of one)
  rr@k        1 / rank of the first relevant chunk in the top k, else 0 (its mean is MRR)
"""

Span = tuple[int, int]


def _spans(items: list[dict], document: str) -> list[Span]:
    return [(i["char_start"], i["char_end"]) for i in items if i["document"] == document]


def is_relevant(chunk: dict, evidence: list[dict]) -> bool:
    start, end = chunk["char_start"], chunk["char_end"]
    return any(s < end and start < e for s, e in _spans(evidence, chunk["document"]))


def hit_at(chunks: list[dict], evidence: list[dict], k: int) -> int:
    return int(any(is_relevant(c, evidence) for c in chunks[:k]))


def reciprocal_rank_at(chunks: list[dict], evidence: list[dict], k: int) -> float:
    for rank, chunk in enumerate(chunks[:k], start=1):
        if is_relevant(chunk, evidence):
            return 1.0 / rank
    return 0.0


def _merge(spans: list[Span]) -> list[Span]:
    merged: list[Span] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def coverage_at(chunks: list[dict], evidence: list[dict], k: int) -> float:
    total = covered = 0
    for document in {e["document"] for e in evidence}:
        wanted = _merge(_spans(evidence, document))
        found = _merge(_spans(chunks[:k], document))
        total += sum(e - s for s, e in wanted)
        covered += sum(max(0, min(e, fe) - max(s, fs)) for s, e in wanted for fs, fe in found)
    return covered / total if total else 0.0
