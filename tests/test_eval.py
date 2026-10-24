import json
import pytest

from baked.evals import EvalPair, run_eval, save_pairs, load_pairs, print_report


class _MockStore:
    """Fake store returning a fixed, ordered result list regardless of query."""

    def __init__(self, results: list[tuple[dict, float]]):
        self._results = results

    def search(self, query, top_k=5, **kwargs):
        return self._results[:top_k]


# ---------------------------------------------------------------------------
# run_eval — recall and MRR
# ---------------------------------------------------------------------------

def test_recall_perfect():
    store = _MockStore([({"title": "A"}, 0.9), ({"title": "B"}, 0.7)])
    result = run_eval(store, [EvalPair("q", {"title": "A"})], top_k=3)
    assert result.recall_at_k == 1.0
    assert result.query_results[0].rank == 1


def test_recall_miss():
    store = _MockStore([({"title": "B"}, 0.9)])
    result = run_eval(store, [EvalPair("q", {"title": "A"})], top_k=3)
    assert result.recall_at_k == 0.0
    assert result.query_results[0].rank is None


def test_recall_found_at_rank_k():
    store = _MockStore([
        ({"title": "B"}, 0.9),
        ({"title": "C"}, 0.8),
        ({"title": "A"}, 0.7),
    ])
    result = run_eval(store, [EvalPair("q", {"title": "A"})], top_k=3)
    assert result.recall_at_k == 1.0
    assert result.query_results[0].rank == 3


def test_recall_beyond_top_k():
    store = _MockStore([
        ({"title": "B"}, 0.9),
        ({"title": "C"}, 0.8),
        ({"title": "A"}, 0.7),
    ])
    result = run_eval(store, [EvalPair("q", {"title": "A"})], top_k=2)
    assert result.recall_at_k == 0.0


def test_mrr_rank_1():
    store = _MockStore([({"title": "A"}, 0.9)])
    result = run_eval(store, [EvalPair("q", {"title": "A"})], top_k=3)
    assert result.mrr == pytest.approx(1.0)


def test_mrr_rank_2():
    store = _MockStore([({"title": "B"}, 0.9), ({"title": "A"}, 0.7)])
    result = run_eval(store, [EvalPair("q", {"title": "A"})], top_k=3)
    assert result.mrr == pytest.approx(0.5)


def test_mrr_averaged_across_pairs():
    store = _MockStore([({"title": "A"}, 0.9), ({"title": "B"}, 0.7)])
    pairs = [
        EvalPair("q1", {"title": "A"}),  # rank 1 → reciprocal 1.0
        EvalPair("q2", {"title": "B"}),  # rank 2 → reciprocal 0.5
    ]
    result = run_eval(store, pairs, top_k=3)
    assert result.mrr == pytest.approx(0.75)


def test_empty_pairs():
    store = _MockStore([])
    result = run_eval(store, [], top_k=5)
    assert result.recall_at_k == 0.0
    assert result.mrr == 0.0
    assert result.n == 0


# ---------------------------------------------------------------------------
# metadata subset matching
# ---------------------------------------------------------------------------

def test_partial_metadata_match():
    store = _MockStore([({"title": "A", "author": "PG", "year": 2004}, 0.9)])
    result = run_eval(store, [EvalPair("q", {"title": "A"})], top_k=3)
    assert result.recall_at_k == 1.0


def test_metadata_mismatch_on_extra_key():
    store = _MockStore([({"title": "A"}, 0.9)])
    result = run_eval(store, [EvalPair("q", {"title": "A", "year": 2004})], top_k=3)
    assert result.recall_at_k == 0.0


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_round_trip(tmp_path):
    pairs = [
        EvalPair(query="What is wealth?", expected={"title": "Wealth"}, label="wealth-q1"),
        EvalPair(query="How to start?", expected={"title": "Start"}),
    ]
    path = str(tmp_path / "pairs.json")
    save_pairs(pairs, path)
    loaded = load_pairs(path)

    assert len(loaded) == 2
    assert loaded[0].query == pairs[0].query
    assert loaded[0].expected == pairs[0].expected
    assert loaded[0].label == pairs[0].label
    assert loaded[1].label == ""


def test_save_load_preserves_unicode(tmp_path):
    pairs = [EvalPair(query="Héllo wörld", expected={"title": "Tëst"})]
    path = str(tmp_path / "pairs.json")
    save_pairs(pairs, path)
    loaded = load_pairs(path)
    assert loaded[0].query == "Héllo wörld"


# ---------------------------------------------------------------------------
# print_report smoke test (no assertion — just must not raise)
# ---------------------------------------------------------------------------

def test_print_report_smoke(capsys):
    store = _MockStore([({"title": "A"}, 0.9), ({"title": "B"}, 0.5)])
    result = run_eval(store, [EvalPair("q", {"title": "A"}, label="test")], top_k=2)
    print_report(result, verbose=True)
    out = capsys.readouterr().out
    assert "Recall@2" in out
    assert "MRR" in out
