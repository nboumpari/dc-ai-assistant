from dc_ai_assistant.rules_assistant.retrieval import RetrievalConfig, all_configs, rrf, tokenize
from evals.rules.retrieval_metrics import coverage_at, hit_at, is_relevant, reciprocal_rank_at


def chunk(start: int, end: int, document: str = "doc") -> dict:
    return {"document": document, "char_start": start, "char_end": end}


EVIDENCE = [{"document": "doc", "char_start": 100, "char_end": 200}]


def test_relevant_needs_overlap_in_the_same_document():
    assert is_relevant(chunk(150, 300), EVIDENCE)
    assert not is_relevant(chunk(0, 100), EVIDENCE)  # touching is not overlapping
    assert not is_relevant(chunk(150, 300, "other"), EVIDENCE)


def test_hit_and_reciprocal_rank_use_the_first_relevant_chunk():
    chunks = [chunk(0, 50), chunk(300, 400), chunk(180, 260), chunk(90, 210)]
    assert hit_at(chunks, EVIDENCE, 2) == 0
    assert hit_at(chunks, EVIDENCE, 3) == 1
    assert reciprocal_rank_at(chunks, EVIDENCE, 5) == 1 / 3
    assert reciprocal_rank_at(chunks, EVIDENCE, 2) == 0.0


def test_coverage_counts_each_evidence_character_once():
    evidence = EVIDENCE + [{"document": "doc", "char_start": 500, "char_end": 600}]
    # first chunk covers 100-150, the overlapping second adds 150-180; nothing of 500-600
    chunks = [chunk(50, 150), chunk(120, 180), chunk(140, 170)]
    assert coverage_at(chunks, evidence, 3) == 80 / 200
    assert coverage_at([chunk(0, 1000)], evidence, 5) == 1.0


def test_rrf_rewards_items_ranked_high_in_several_lists():
    fused = rrf([[1, 2, 3], [3, 4, 1]], k=60)
    assert fused[:2] == [1, 3]  # in both lists, ranks (1, 3) and (3, 1)
    assert set(fused) == {1, 2, 3, 4}


def test_tokenize_keeps_section_numbers_together():
    assert tokenize("See 1.1.3.6 for UN/ECE rules") == ["see", "1.1.3.6", "for", "un/ece", "rules"]


def test_the_grid_has_45_distinct_configs():
    configs = all_configs()
    assert len(configs) == len({c.name for c in configs}) == 45
    assert RetrievalConfig("fixed", "bm25").name == "fixed-bm25-norerank"
