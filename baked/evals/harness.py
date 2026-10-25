from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class EvalPair:
    """A single labeled evaluation example.

    query:    the question or search string to issue against the store.
    expected: a metadata subset that the correct result must contain.
              Partial match — {"title": "X"} matches any result whose metadata
              has at least title="X", regardless of other keys.
    label:    optional human-readable name shown in print_report output.
    """
    query: str
    expected: dict[str, Any]
    label: str = ""


@dataclass
class QueryResult:
    """Outcome of a single query during an eval run."""
    pair: EvalPair
    rank: int | None  # 1-indexed rank of first match; None = not found in top_k
    hits: list[dict[str, Any]]  # metadata of each top_k result
    scores: list[float]  # cosine scores of each top_k result


@dataclass
class EvalResult:
    """Aggregate metrics from run_eval().

    recall_at_k: fraction of queries where the correct doc appeared in top-k.
    mrr:         mean reciprocal rank across all queries.
    top_k:       the k used during this eval run.
    n:           total number of queries evaluated.
    query_results: per-query breakdown for verbose inspection.
    """
    recall_at_k: float
    mrr: float
    top_k: int
    n: int
    query_results: list[QueryResult]

    def __str__(self) -> str:
        return (
            f"Recall@{self.top_k}: {self.recall_at_k:.3f}  "
            f"MRR: {self.mrr:.3f}  "
            f"(n={self.n})"
        )


def _matches(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(metadata.get(k) == v for k, v in expected.items())


def run_eval(
    store: Any,
    pairs: list[EvalPair],
    top_k: int = 5,
) -> EvalResult:
    """Evaluate retrieval quality against labeled query-document pairs.

    Works with EmbeddingStore (pass str queries in EvalPair.query) and
    VectorStore (pass vector queries). Any object with a compatible
    .search(query, top_k) interface works.

    Args:
        store:  the store to evaluate. Must implement .search(query, top_k).
        pairs:  labeled examples — see EvalPair.
        top_k:  number of results to retrieve per query.

    Returns:
        EvalResult with Recall@k, MRR, and per-query breakdown.
    """
    query_results: list[QueryResult] = []

    for pair in pairs:
        if not pair.expected:
            raise ValueError(
                f"EvalPair has empty expected dict — every result would match. "
                f"Set at least one key-value pair. (query={pair.query!r})"
            )
        results = store.search(pair.query, top_k=top_k)
        hits = [meta for meta, _ in results]
        scores = [score for _, score in results]
        rank = next(
            (i for i, meta in enumerate(hits, 1) if _matches(meta, pair.expected)),
            None,
        )
        query_results.append(
            QueryResult(pair=pair, rank=rank, hits=hits, scores=scores)
        )

    n = len(pairs)
    recall = sum(1 for r in query_results if r.rank is not None) / n if n else 0.0
    mrr = sum(1.0 / r.rank for r in query_results if r.rank is not None) / n if n else 0.0

    return EvalResult(
        recall_at_k=recall,
        mrr=mrr,
        top_k=top_k,
        n=n,
        query_results=query_results,
    )


def print_report(result: EvalResult, verbose: bool = False) -> None:
    """Print a human-readable summary of an EvalResult.

    Args:
        result:  output from run_eval().
        verbose: if True, prints rank and label for every query.
                 Missed queries also show the top-scoring result for debugging.
    """
    print(str(result))
    if verbose:
        for qr in result.query_results:
            status = f"rank {qr.rank}" if qr.rank is not None else "MISS"
            label = qr.pair.label or qr.pair.query[:70]
            print(f"  [{status:>6}] {label!r}")
            if qr.rank is None and qr.hits:
                print(f"           top hit: {qr.hits[0]}")


def save_pairs(pairs: list[EvalPair], path: str) -> None:
    """Save eval pairs to a JSON file so they can be reused across runs.

    Generate pairs once (with generate_pairs or by hand), save them,
    then load and re-run whenever you change embeddings or search params.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in pairs], f, indent=2, ensure_ascii=False)


def load_pairs(path: str) -> list[EvalPair]:
    """Load eval pairs previously saved with save_pairs()."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [EvalPair(**d) for d in data]
