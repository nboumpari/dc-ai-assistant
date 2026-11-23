"""Generate the rules agent's golden set (reviewed by a person before use).

Run from the project root, after extract:

    uv run python -m evals.rules.generate_golden_set                 # all extracted documents
    uv run python -m evals.rules.generate_golden_set adr_2025        # add one document later
    uv run python -m evals.rules.generate_golden_set --replace R36   # replace questions found weak in review

Writes evals/rules/golden_set.yaml. Documents already in the set are kept
unless named on the command line, so a document can be added later without
regenerating the rest. --replace keeps the question's id and split.

How questions are made (following current research on RAG test sets):
- Questions are generated from whole document sections, never from chunks,
  so no chunking strategy is favoured.
- A stronger model (GOLDEN_MODEL) than the one being evaluated drafts each
  question with verbatim evidence quotes and a reference answer.
- Automatic checks: every quote must be found word for word in the document
  (this gives its character span); the question must not copy long phrases
  from the evidence (that would favour keyword search); no near-duplicates.
- "not_found" questions are checked against the corpus: the closest passages
  are retrieved and a model confirms they do not answer the question, not
  even in part.
- Out-of-scope questions (operations data, off-topic) expect "decline".
A person reviews every question before the set is used; weak ones are
replaced with --replace.
"""

import json
import random
import re
import sys
from pathlib import Path

import yaml
from rank_bm25 import BM25Okapi

from dc_ai_assistant import settings  # noqa: F401  (loads .env)

ROOT = Path(__file__).resolve().parents[2]
TEXT_DIR = ROOT / "data" / "documents" / "text"
CHUNK_DIR = ROOT / "data" / "documents" / "chunks"
GOLDEN_PATH = Path(__file__).with_name("golden_set.yaml")
CORPUS_PATH = ROOT / "scripts" / "rules_corpus" / "corpus.yaml"
CORPUS_SECTIONS = {  # documents that use only selected sections (GS1 General Specifications)
    d["id"]: d["sections"]
    for d in yaml.safe_load(CORPUS_PATH.read_text(encoding="utf-8"))["documents"]
    if d.get("sections")
}

GOLDEN_MODEL = "gpt-5.5"
SEED = 42
PER_DOCUMENT = 7  # including one multi-section question where the document allows it
NOT_FOUND = 7
OUT_OF_SCOPE = 4
TARGET_TOTAL = 60  # 7 documents x 7 + 7 not found + 4 out of scope
MIN_QUOTE_CHARS = 40  # an evidence quote must be at least a real sentence, not a lead-in fragment
TEST_SHARE = 0.3
SECTION_MIN, SECTION_MAX = 500, 6000
MAX_COPIED_WORDS = 6  # a question may not repeat 6 or more consecutive words of its evidence
QUESTION_TYPES = ["definition", "rule or limit", "procedure", "table lookup", "rule or limit", "definition"]

QUESTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["usable", "question", "question_type", "difficulty", "evidence_quotes", "reference_answer"],
    "properties": {
        "usable": {
            "type": "boolean",
            "description": "false if the passage has no substantive rule or fact to ask about",
        },
        "question": {"type": "string"},
        "question_type": {
            "type": "string",
            "enum": ["definition", "rule or limit", "procedure", "table lookup", "multi-section"],
        },
        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
        "evidence_quotes": {"type": "array", "items": {"type": "string"}},
        "reference_answer": {"type": "string"},
    },
}

GENERATION_PROMPT = """You write evaluation questions for an assistant used by the warehouse and transport \
planners of an FMCG manufacturer (consumer goods shipped on pallets by road from a plant and distribution \
centres to retailers). The assistant answers from official documents.

Document: {title}
Section: {section}

Passage:
\"\"\"
{passage}
\"\"\"

Write ONE question of type "{qtype}" that this passage answers. If that type does not fit the passage, \
choose the type that fits best.

Rules:
- Ask it the way a busy warehouse or transport planner would: short, practical, in everyday words. \
Do not copy phrases from the passage and do not mention section or article numbers.
- The passage alone must fully answer it, and the answer must be specific: a number, limit, condition, \
named item or concrete step. Avoid broad questions that many parts of the document could answer equally well.
- Only ask what such a planner would realistically need: goods transport, labelling, warehouse work and \
safety, dangerous goods, a manufacturer's product safety duties. Not passenger transport, not the internal \
procedures of authorities or legislators, not retail checkout details, not other industries.
- evidence_quotes: 1 to 3 excerpts copied EXACTLY, character for character, from the passage \
(without any Markdown symbols such as ** or #). Together they must contain EVERY fact stated in the \
reference answer. Each quote is a full sentence, list item or table row, never a lead-in that ends with a colon.
- reference_answer: a correct, concise answer (1-3 sentences) based only on the passage.
- difficulty: easy (one stated fact), medium (needs some reading), hard (combines several facts).
- Set usable to false if the passage is only a table of contents, a heading list, contributors, \
change log or similar, with no rule or fact worth asking about."""


REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["relevant", "specific", "evidence_complete", "answer_correct", "reason"],
    "properties": {
        "relevant": {"type": "boolean"},
        "specific": {"type": "boolean"},
        "evidence_complete": {"type": "boolean"},
        "answer_correct": {"type": "boolean"},
        "reason": {"type": "string"},
    },
}

REVIEW_PROMPT = """Review a draft evaluation question for an assistant used by the warehouse and transport \
planners of an FMCG manufacturer (consumer goods on pallets, road transport, plant and distribution centres).

Passage:
\"\"\"
{passage}
\"\"\"

Question: {question}
Evidence quotes: {quotes}
Reference answer: {answer}

Answer strictly:
- relevant: would such a planner realistically ask this at work? (false for passenger transport, internal \
procedures of authorities or legislators, retail checkout details, other industries)
- specific: is the answer particular to this passage (a number, limit, condition, named item or step), \
rather than general advice that many passages could give?
- evidence_complete: do the evidence quotes ALONE contain every fact in the reference answer?
- answer_correct: is the reference answer correct according to the passage?
- reason: one short sentence."""


# ---------------------------------------------------------------- OpenAI


def _client():
    from langsmith.wrappers import wrap_openai
    from openai import OpenAI

    return wrap_openai(OpenAI())


def ask_json(client, prompt: str, schema: dict, name: str) -> dict:
    response = client.responses.create(
        model=GOLDEN_MODEL,
        input=prompt,
        reasoning={"effort": "medium"},
        text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
    )
    return json.loads(response.output_text)


# ---------------------------------------------------------------- checks


def _normalise_with_map(text: str) -> tuple[str, list[int]]:
    """Lower-case text without Markdown emphasis and with single spaces, plus a map back to original indexes."""
    out, index, last_space = [], [], False
    for i, ch in enumerate(text):
        if ch in "*_#`|":
            continue
        if ch.isspace():
            if last_space:
                continue
            out.append(" ")
            index.append(i)
            last_space = True
            continue
        out.append(ch.lower())
        index.append(i)
        last_space = False
    return "".join(out), index


def locate(quote: str, text: str, norm: tuple[str, list[int]]) -> tuple[int, int] | None:
    """Character span of a quote in the document text, ignoring Markdown symbols and spacing."""
    q, _ = _normalise_with_map(quote)
    q = q.strip()
    if len(q) < 15:
        return None
    normalised, index = norm
    pos = normalised.find(q)
    if pos == -1:
        return None
    return index[pos], index[pos + len(q) - 1] + 1


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def copied_words(question: str, evidence: str) -> int:
    """Length of the longest run of consecutive words the question shares with the evidence."""
    q, e = _words(question), " " + " ".join(_words(evidence)) + " "
    best = 0
    for i in range(len(q)):
        for j in range(i + best + 1, len(q) + 1):
            if " " + " ".join(q[i:j]) + " " in e:
                best = j - i
            else:
                break
    return best


def too_similar(question: str, others: list[str]) -> bool:
    words = set(_words(question))
    for other in others:
        o = set(_words(other))
        if words and o and len(words & o) / len(words | o) > 0.6:
            return True
    return False


# ---------------------------------------------------------------- answerable questions


FRONT_MATTER = re.compile(r"contents|contributors|log of changes|disclaimer|document summary|references|further", re.I)
# Institutional provisions a planner would never ask about: treaty signature, EU law-making, reviews, reporting
ADMINISTRATIVE = re.compile(
    r"signature|accession|accede|ratif|denunciat|depositary|reservation|secretary-general|"
    r"delegated act|implementing act|exercise of the delegation|committee procedure|"
    r"shall carry out an evaluation|shall (?:submit|present) a report|communicate to the commission|"
    r"shall enter into force|is (?:hereby )?(?:amended|repealed)|correlation table|transitional provision",
    re.I,
)


RECITAL = re.compile(r"^[-*\s]*\(\d{1,3}\)\s")


def _is_recitals(body: str) -> bool:
    """EU regulations open with numbered recitals '(12) ...': explanatory, not binding, so not asked about."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    return bool(paragraphs) and sum(bool(RECITAL.match(p)) for p in paragraphs) / len(paragraphs) > 0.5


def _windows(text: str, section: dict) -> list[dict]:
    """Split a long section into passages of at most SECTION_MAX characters at paragraph boundaries."""
    start, end = section["start"], section["end"]
    if end - start <= SECTION_MAX:
        return [section]
    boundaries = [start + m.end() for m in re.finditer(r"\n\s*\n", text[start:end])] + [end]
    windows, window_start, last_cut = [], start, start
    for boundary in boundaries:
        if boundary - window_start > SECTION_MAX and last_cut > window_start:
            windows.append((window_start, last_cut))
            window_start = last_cut
        last_cut = boundary
    windows.append((window_start, end))
    return [{**section, "start": a, "end": b} for a, b in windows if b - a >= SECTION_MIN]


def candidate_sections(text: str, sections: list[dict], selected: list[str] | None = None) -> list[dict]:
    """Operational passages to ask about: no front matter, no institutional provisions, long sections split
    into windows, and for documents with a section selection, only the selected sections."""
    out = []
    for s in sections:
        body = text[s["start"] : s["end"]]
        if len(body) < SECTION_MIN or FRONT_MATTER.search(s["title"]):
            continue
        if ADMINISTRATIVE.search(body[:600]) or _is_recitals(body):
            continue
        if selected and not any(s["id"] == n or s["id"].startswith(n + ".") for n in selected):
            continue
        out.extend(_windows(text, s))
    return out


def _parent(section_id: str) -> str:
    """'3.4.2' -> '3.4'; articles share the parent 'Article'."""
    return "Article" if section_id.startswith("Article ") else section_id.rsplit(".", 1)[0]


def spread(items: list, k: int, rng: random.Random) -> list:
    """k items spread over the whole document (one from each of k equal slices)."""
    if len(items) <= k:
        return items
    size = len(items) / k
    return [items[int(i * size) + rng.randrange(max(1, int(size)))] for i in range(k)]


def _overlaps(spans: list[tuple[int, int]], taken: list[tuple[int, int]]) -> bool:
    return any(a < d and c < b for a, b in spans for c, d in taken)


def generate_for_document(
    client,
    doc_id: str,
    rng: random.Random,
    existing: list[str],
    target: int,
    used: set[tuple[int, int]],
    taken_spans: list[tuple[int, int]],
    allow_multi: bool = True,
) -> list[dict]:
    """Generate up to `target` accepted questions, trying passages until enough pass every check.
    `used` (passage ranges already tried) and `taken_spans` (accepted evidence) are updated in place."""
    text = (TEXT_DIR / f"{doc_id}.md").read_text(encoding="utf-8")
    meta = json.loads((TEXT_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))
    norm = _normalise_with_map(text)
    candidates = candidate_sections(text, meta["sections"], CORPUS_SECTIONS.get(doc_id))
    fresh = [c for c in candidates if (c["start"], c["end"]) not in used]
    first = spread(fresh, target * 2, rng)
    rng.shuffle(first)
    rest = [c for c in fresh if c not in first]
    rng.shuffle(rest)

    questions: list[dict] = []
    jobs = [(s, QUESTION_TYPES[i % len(QUESTION_TYPES)]) for i, s in enumerate(first + rest)]
    # one multi-section question: two consecutive passages with the same parent section
    pairs = [
        (a, b)
        for a, b in zip(candidates, candidates[1:], strict=False)
        if _parent(a["id"]) == _parent(b["id"]) and a["id"] != b["id"]
    ]
    if pairs and allow_multi:
        jobs.insert(0, (rng.choice(pairs), "multi-section"))

    for section, qtype in jobs:
        if len(questions) >= target:
            break
        if qtype == "multi-section" and any(q["question_type"] == "multi-section" for q in questions):
            continue
        parts = section if isinstance(section, tuple) else (section,)
        for s in parts:
            used.add((s["start"], s["end"]))
        passage = "\n\n".join(text[s["start"] : s["end"]] for s in parts)
        label = " + ".join(s["title"] for s in parts)
        draft = ask_json(
            client,
            GENERATION_PROMPT.format(title=meta["title"], section=label, passage=passage, qtype=qtype),
            QUESTION_SCHEMA,
            "golden_question",
        )
        if not draft["usable"]:
            continue
        spans = [locate(q, text, norm) for q in draft["evidence_quotes"]]
        if not spans or None in spans:
            print(f"    rejected (quote not found verbatim): {draft['question'][:70]}")
            continue
        quotes = [text[a:b].strip() for a, b in spans]
        if any(len(q) < MIN_QUOTE_CHARS or q.endswith(":") for q in quotes):
            print(f"    rejected (evidence is a fragment or lead-in): {draft['question'][:70]}")
            continue
        evidence_text = " ".join(quotes)
        if copied_words(draft["question"], evidence_text) >= MAX_COPIED_WORDS:
            print(f"    rejected (copies the evidence wording): {draft['question'][:70]}")
            continue
        if too_similar(draft["question"], existing + [q["question"] for q in questions]):
            print(f"    rejected (near-duplicate): {draft['question'][:70]}")
            continue
        if _overlaps(spans, taken_spans):
            print(f"    rejected (same evidence as another question): {draft['question'][:70]}")
            continue
        review = ask_json(
            client,
            REVIEW_PROMPT.format(
                passage=passage,
                question=draft["question"],
                quotes=json.dumps(quotes, ensure_ascii=False),
                answer=draft["reference_answer"],
            ),
            REVIEW_SCHEMA,
            "question_review",
        )
        failed = [k for k in ("relevant", "specific", "evidence_complete", "answer_correct") if not review[k]]
        if failed:
            print(f"    rejected (review: {', '.join(failed)}): {draft['question'][:60]}")
            continue
        taken_spans.extend(spans)
        pages = sorted(
            {p["page"] for a, b in spans for p in meta["pages"] if p["page"] and p["start"] < b and p["end"] > a}
        )
        questions.append(
            {
                "question": draft["question"],
                "behavior": "answer",
                "route": "rules_agent",
                "question_type": "multi-section" if len(parts) > 1 else draft["question_type"],
                "difficulty": draft["difficulty"],
                "document": doc_id,
                "sections": [s["id"] for s in parts],
                "pages": pages,
                "evidence": [{"char_start": a, "char_end": b, "quote": text[a:b]} for a, b in spans],
                "reference_answer": draft["reference_answer"],
            }
        )
        print(f"    ok  [{questions[-1]['question_type']}] {draft['question'][:80]}")
    return questions


# ---------------------------------------------------------------- not found and out of scope

NOT_FOUND_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["questions"],
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
}

NOT_FOUND_CHECK_PASSAGES = 10  # closest BM25 passages shown to the not-found check

VERIFY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answered"],
    "properties": {"answered": {"type": "boolean"}},
}


def _bm25_index():
    chunks = [json.loads(line) for line in (CHUNK_DIR / "structure.jsonl").read_text(encoding="utf-8").splitlines()]
    return chunks, BM25Okapi([_words(c["text"]) for c in chunks])


def generate_not_found(
    client, titles: list[str], existing: list[str], needed: int = NOT_FOUND, rounds: int = 4
) -> list[dict]:
    """Questions the corpus does not answer; generated in rounds until `needed` pass the corpus check."""
    chunks, bm25 = _bm25_index()
    kept: list[dict] = []
    for _ in range(rounds):
        if len(kept) >= needed:
            break
        avoid = "\n- ".join(existing + [k["question"] for k in kept]) or "(none)"
        prompt = (
            "These official documents are available to a warehouse and transport planning assistant:\n- "
            + "\n- ".join(titles)
            + f"\n\nWrite {NOT_FOUND * 2} realistic, short questions a warehouse or transport planner of an FMCG "
            "manufacturer might ask about rules, standards or regulations, which these documents do NOT answer "
            "(for example other countries' rules, fees, topics outside their scope). Use everyday wording, and do "
            f"not repeat these questions:\n- {avoid}"
        )
        kept += _check_not_found(
            client,
            ask_json(client, prompt, NOT_FOUND_SCHEMA, "not_found")["questions"],
            chunks,
            bm25,
            existing + [k["question"] for k in kept],
            needed - len(kept),
        )
    return kept


def _check_not_found(client, candidates: list[str], chunks, bm25, existing: list[str], needed: int) -> list[dict]:
    kept = []
    for question in candidates:
        if len(kept) >= needed or too_similar(question, existing + [k["question"] for k in kept]):
            continue
        scores = bm25.get_scores(_words(question))
        top = sorted(range(len(chunks)), key=lambda i: -scores[i])[:NOT_FOUND_CHECK_PASSAGES]
        passages = "\n\n---\n\n".join(chunks[i]["text"][:1500] for i in top)
        verdict = ask_json(
            client,
            f"Question: {question}\n\nPassages:\n{passages}\n\n"
            "Do these passages answer the question fully or in part? Answer true if any passage contains "
            "information a careful assistant could cite as even a partial answer.",
            VERIFY_SCHEMA,
            "not_found_check",
        )
        if verdict["answered"]:
            print(f"    rejected (the corpus answers it fully or partly): {question[:70]}")
            continue
        kept.append(
            {"question": question, "behavior": "not_found", "route": "rules_agent", "question_type": "not in corpus"}
        )
        print(f"    ok  [not in corpus] {question[:80]}")
    return kept


def generate_out_of_scope(client) -> list[dict]:
    prompt = (
        "A company has two assistants: one answers questions about its own 2025 warehouse operations data "
        "(orders, shipments, pallets, carriers, dock appointments, KPIs such as OTIF), the other answers questions "
        "about official rules (GS1 labelling, road transport law, warehouse safety, dangerous goods, product safety). "
        f"Write {OUT_OF_SCOPE // 2} short questions for the operations-data assistant and {OUT_OF_SCOPE // 2} short "
        "questions that neither assistant should answer (unrelated to warehousing). "
        "Return the operations-data ones first."
    )
    questions = ask_json(client, prompt, NOT_FOUND_SCHEMA, "out_of_scope_questions")["questions"][:OUT_OF_SCOPE]
    half = OUT_OF_SCOPE // 2
    out = []
    for i, question in enumerate(questions):
        route = "data_agent" if i < half else "out_of_scope"
        out.append({"question": question, "behavior": "decline", "route": route, "question_type": "out of scope"})
        print(f"    ok  [out of scope -> {route}] {question[:80]}")
    return out


# ---------------------------------------------------------------- assembling


def assign_splits(questions: list[dict], rng: random.Random) -> None:
    """Stratified dev/test split: about 30% test within each document / question group."""
    groups: dict[str, list[dict]] = {}
    for q in questions:
        groups.setdefault(q.get("document") or q["question_type"], []).append(q)
    for group in groups.values():
        rng.shuffle(group)
        n_test = max(1, round(len(group) * TEST_SHARE))
        for i, q in enumerate(group):
            q["split"] = "test" if i < n_test else "dev"


GOLDEN_HEADER = (
    "# Golden set for the rules agent - generated by evals/rules/generate_golden_set.py,\n"
    "# then reviewed by a person: realistic question, right evidence, correct reference answer.\n"
    "# Upload with: uv run python -m evals.rules.upload_golden_set\n\n"
)


def replace_questions(ids: list[str]) -> int:
    """Replace the given questions, keeping id and split.

    Answerable questions get a new question from the same document; not-found questions get a new
    not-found question that passes the corpus check.
    """
    questions = yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8"))["questions"]
    by_id = {q["id"]: q for q in questions}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        print(f"Unknown ids: {', '.join(unknown)}", file=sys.stderr)
        return 1
    client = _client()
    texts = [q["question"] for q in questions]
    failed = [i for i in ids if by_id[i]["behavior"] not in ("answer", "not_found")]
    not_found_ids = [i for i in ids if by_id[i]["behavior"] == "not_found"]
    if not_found_ids:
        print(f"  replacing {', '.join(not_found_ids)} (not in corpus)")
        documents = sorted(p.stem for p in TEXT_DIR.glob("*.md"))
        titles = [json.loads((TEXT_DIR / f"{d}.json").read_text(encoding="utf-8"))["title"] for d in documents]
        new = generate_not_found(client, titles, texts, needed=len(not_found_ids))
        for qid, q in zip(not_found_ids, new, strict=False):  # fewer may pass the check
            texts.append(q["question"])
            by_id[qid] = {"id": qid, **q, "split": by_id[qid]["split"]}
        failed += not_found_ids[len(new) :]
    for qid in ids:
        old = by_id[qid]
        if old["behavior"] != "answer":
            continue
        doc_id = old["document"]
        taken = [
            (e["char_start"], e["char_end"]) for q in questions if q.get("document") == doc_id for e in q["evidence"]
        ]
        print(f"  replacing {qid} ({doc_id})")
        new = generate_for_document(
            client, doc_id, random.Random(f"{SEED}-{doc_id}-{qid}"), texts, 1, set(), taken, allow_multi=False
        )
        if not new:
            failed.append(qid)
            continue
        texts.append(new[0]["question"])
        by_id[qid] = {"id": qid, **new[0], "split": old["split"]}
    ordered = [by_id[q["id"]] for q in questions]
    GOLDEN_PATH.write_text(
        GOLDEN_HEADER + yaml.safe_dump({"questions": ordered}, sort_keys=False, allow_unicode=True, width=110),
        encoding="utf-8",
    )
    done = [i for i in ids if i not in failed]
    print(f"\nReplaced {len(done)} question(s): {', '.join(done) or '-'}")
    if failed:
        print(f"Could not replace: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    if sys.argv[1:2] == ["--replace"]:
        return replace_questions(sys.argv[2:])
    only = set(sys.argv[1:])
    rng = random.Random(SEED)
    documents = sorted(p.stem for p in TEXT_DIR.glob("*.md"))
    if not documents:
        print("No extracted documents; run scripts.rules_corpus.extract first.", file=sys.stderr)
        return 1

    previous = yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8"))["questions"] if GOLDEN_PATH.exists() else []
    keep = [q for q in previous if q.get("document") and q["document"] not in only and q["document"] in documents]
    questions = list(keep)
    client = _client()

    todo = [
        d
        for d in documents
        if not (only and d not in only) and not (not only and any(q.get("document") == d for q in keep))
    ]
    used: dict[str, set] = {d: set() for d in todo}
    taken: dict[str, list] = {
        d: [(e["char_start"], e["char_end"]) for q in keep if q.get("document") == d for e in q["evidence"]]
        for d in todo
    }
    rngs = {d: random.Random(f"{SEED}-{d}") for d in todo}
    for doc_id in todo:
        print(f"  {doc_id}")
        questions += generate_for_document(
            client, doc_id, rngs[doc_id], [q["question"] for q in questions], PER_DOCUMENT, used[doc_id], taken[doc_id]
        )
    # top up: if a document ran short, take extra questions from documents that still have unused passages
    answer_target = len(todo) * PER_DOCUMENT + sum(1 for q in keep)
    for _ in range(3):
        short = answer_target - sum(1 for q in questions if q["behavior"] == "answer")
        if short <= 0:
            break
        print(f"  topping up {short} question(s)")
        for doc_id in sorted(todo, key=lambda d: sum(1 for q in questions if q.get("document") == d)):
            if short <= 0:
                break
            extra = generate_for_document(
                client,
                doc_id,
                rngs[doc_id],
                [q["question"] for q in questions],
                1,
                used[doc_id],
                taken[doc_id],
                allow_multi=False,
            )
            questions += extra
            short -= len(extra)

    others = [q for q in previous if not q.get("document")]
    if others and only:
        questions += others
    else:
        titles = [json.loads((TEXT_DIR / f"{d}.json").read_text(encoding="utf-8"))["title"] for d in documents]
        print("  not in corpus")
        questions += generate_not_found(client, titles, [q["question"] for q in questions])
        print("  out of scope")
        questions += generate_out_of_scope(client)

    assign_splits(questions, rng)
    order = {d: i for i, d in enumerate(documents)}
    questions.sort(key=lambda q: (order.get(q.get("document"), len(order)), q["question_type"], q["question"]))
    for i, q in enumerate(questions, start=1):
        q["id"] = f"R{i:02d}"
    ordered = [{"id": q.pop("id"), **q} for q in questions]

    header = (
        "# DRAFT golden set for the rules agent - generated, to be reviewed by a person before use.\n"
        "# Check each question: is it realistic, is the evidence right, is the reference answer correct?\n"
        "# Delete questions that are wrong; small wording fixes are fine.\n\n"
    )
    GOLDEN_PATH.write_text(
        header + yaml.safe_dump({"questions": ordered}, sort_keys=False, allow_unicode=True, width=110),
        encoding="utf-8",
    )
    by_split = {s: sum(1 for q in ordered if q["split"] == s) for s in ("dev", "test")}
    print(f"\nWrote {len(ordered)} questions ({by_split['dev']} dev, {by_split['test']} test) to {GOLDEN_PATH.name}")
    if not only and len(ordered) != TARGET_TOTAL:
        print(f"WARNING: expected {TARGET_TOTAL} questions, got {len(ordered)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
