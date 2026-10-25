import json
import math
import re
import struct
import os
import warnings
from typing import Any

_HEADER_FMT = "II"  # dim, count — both unsigned int
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

_BM25_K1 = 1.5
_BM25_B = 0.75


def _normalize(vec: list[float]) -> list[float]:
    if not all(math.isfinite(x) for x in vec):
        raise ValueError("Vector contains NaN or infinite values.")
    magnitude = math.sqrt(sum(x * x for x in vec))
    if magnitude == 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return [x / magnitude for x in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _minmax(scores: list[float]) -> list[float]:
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [0.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def _bm25(query_tokens: list[str], corpus: list[list[str]], deleted: set[int]) -> list[float]:
    active = [i for i in range(len(corpus)) if i not in deleted]
    if not active or not query_tokens:
        return [0.0] * len(corpus)

    avgdl = sum(len(corpus[i]) for i in active) / len(active)

    df: dict[str, int] = {}
    for i in active:
        for term in set(corpus[i]):
            df[term] = df.get(term, 0) + 1

    n = len(active)
    scores = []
    for i, doc in enumerate(corpus):
        if i in deleted:
            scores.append(0.0)
            continue
        dl = len(doc)
        tf_map: dict[str, int] = {}
        for term in doc:
            tf_map[term] = tf_map.get(term, 0) + 1
        score = 0.0
        for term in query_tokens:
            if term not in df:
                continue
            idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1)
            tf = tf_map.get(term, 0)
            score += idf * (tf * (_BM25_K1 + 1)) / (tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avgdl))
        scores.append(score)

    return scores


class VectorStore:
    def __init__(self, path: str, text_field: str = "text"):
        self._path = path
        self._vecs_path = path + ".vecs"
        self._meta_path = path + ".json"
        self._text_field = text_field
        self._dim: int | None = None
        self._vectors: list[list[float]] = []
        self._metadata: list[dict[str, Any]] = []
        self._corpus: list[list[str]] = []
        self._deleted: set[int] = set()

        if os.path.exists(self._vecs_path) and os.path.exists(self._meta_path):
            self._load()

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def add(self, embedding: list[float], metadata: dict[str, Any] | None = None) -> int:
        """Add a vector with optional metadata. Returns its id."""
        if self._dim is not None and len(embedding) != self._dim:
            raise ValueError(f"Dimension mismatch: expected {self._dim}, got {len(embedding)}.")

        normalized = _normalize(embedding)  # raises on zero/nan/inf
        meta = dict(metadata or {})
        tokens = _tokenize(str(meta.get(self._text_field, "")))

        if self._dim is None:
            self._dim = len(embedding)

        idx = len(self._vectors)
        self._vectors.append(normalized)
        self._metadata.append(meta)
        self._corpus.append(tokens)
        return idx

    def add_many(self, embeddings: list[list[float]], metadatas: list[dict[str, Any]] | None = None) -> list[int]:
        """Add multiple vectors at once. Returns list of assigned ids."""
        if metadatas is None:
            metadatas = [{} for _ in embeddings]
        if len(embeddings) != len(metadatas):
            raise ValueError(
                f"Length mismatch: {len(embeddings)} embeddings but {len(metadatas)} metadatas."
            )
        return [self.add(emb, meta) for emb, meta in zip(embeddings, metadatas)]

    def remove(self, id: int) -> None:
        """Soft-delete a vector by id. Call save() to persist."""
        if id < 0 or id >= len(self._vectors):
            raise IndexError(f"Id {id} out of range.")
        self._deleted.add(id)

    def update(
        self,
        id: int,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update vector and/or metadata for an existing id. At least one must be provided."""
        if embedding is None and metadata is None:
            raise ValueError("At least one of embedding or metadata must be provided.")
        if id < 0 or id >= len(self._vectors) or id in self._deleted:
            raise IndexError(f"Id {id} is out of range or has been deleted.")
        if embedding is not None:
            if self._dim is not None and len(embedding) != self._dim:
                raise ValueError(f"Dimension mismatch: expected {self._dim}, got {len(embedding)}.")
            self._vectors[id] = _normalize(embedding)
        if metadata is not None:
            self._metadata[id] = dict(metadata)
            self._corpus[id] = _tokenize(str(metadata.get(self._text_field, "")))

    def save(self) -> None:
        """Flush all data to disk, compacting deleted entries."""
        active_ids = [i for i in range(len(self._vectors)) if i not in self._deleted]

        if not active_ids:
            for p in (self._vecs_path, self._meta_path):
                if os.path.exists(p):
                    os.remove(p)
            self._vectors = []
            self._metadata = []
            self._corpus = []
            self._deleted = set()
            return

        dim = self._dim
        count = len(active_ids)
        vec_fmt = f"{dim}f"

        # Serialize JSON first — catches non-serializable metadata before touching files
        meta_out = [self._metadata[i] for i in active_ids]
        meta_json = json.dumps(meta_out)

        vecs_tmp = self._vecs_path + ".tmp"
        meta_tmp = self._meta_path + ".tmp"
        try:
            with open(vecs_tmp, "wb") as f:
                f.write(struct.pack(_HEADER_FMT, dim, count))
                for i in active_ids:
                    f.write(struct.pack(vec_fmt, *self._vectors[i]))

            with open(meta_tmp, "w", encoding="utf-8") as f:
                f.write(meta_json)

            os.replace(vecs_tmp, self._vecs_path)
            os.replace(meta_tmp, self._meta_path)
        except Exception:
            for p in (vecs_tmp, meta_tmp):
                if os.path.exists(p):
                    os.remove(p)
            raise

        # Compact in-memory state to match saved state
        self._vectors = [self._vectors[i] for i in active_ids]
        self._metadata = meta_out
        self._corpus = [self._corpus[i] for i in active_ids]
        self._deleted = set()

    # ------------------------------------------------------------------
    # Search API
    # ------------------------------------------------------------------

    def search(
        self,
        query: list[float],
        top_k: int = 5,
        radius: float | None = None,
        filter: dict[str, Any] | None = None,
        text_query: str | None = None,
        alpha: float = 0.5,
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Return up to top_k results as [(metadata, score)], highest score first.
        If radius is set, return all results with score >= radius (top_k ignored).
        filter: only consider vectors whose metadata contains all key-value pairs.
        text_query: when provided, hybrid search merges BM25 and semantic scores.
        alpha: weight for semantic score (1.0 = pure semantic, 0.0 = pure BM25).
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
        if not self._vectors:
            return []
        if self._dim is not None and len(query) != self._dim:
            raise ValueError(f"Query dimension mismatch: expected {self._dim}, got {len(query)}.")

        if text_query is not None and alpha == 1.0:
            warnings.warn("text_query has no effect when alpha=1.0; use alpha < 1.0 for hybrid search.")

        q = _normalize(query)
        active = [i for i in range(len(self._vectors)) if i not in self._deleted]

        if not active:
            return []

        sem_scores = [_dot(q, self._vectors[i]) for i in active]

        if text_query is not None and alpha < 1.0:
            bm25_raw = _bm25(_tokenize(text_query), self._corpus, self._deleted)
            bm25_scores = [bm25_raw[i] for i in active]
            sem_norm = _minmax(sem_scores)
            bm25_norm = _minmax(bm25_scores)
            final_scores = [alpha * s + (1.0 - alpha) * b for s, b in zip(sem_norm, bm25_norm)]
        else:
            final_scores = sem_scores

        results = []
        for i, score in zip(active, final_scores):
            if filter and not _matches(self._metadata[i], filter):
                continue
            results.append((self._metadata[i], score))

        if radius is not None:
            results = [(m, s) for m, s in results if s >= radius]
            results.sort(key=lambda x: x[1], reverse=True)
        else:
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:top_k]

        return results

    def search_many(
        self,
        queries: list[list[float]],
        top_k: int = 5,
        radius: float | None = None,
        filter: dict[str, Any] | None = None,
        text_query: str | None = None,
        alpha: float = 0.5,
    ) -> list[list[tuple[dict[str, Any], float]]]:
        """Batch search. Returns a list of result lists, one per query."""
        return [
            self.search(q, top_k=top_k, radius=radius, filter=filter, text_query=text_query, alpha=alpha)
            for q in queries
        ]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._vectors) - len(self._deleted)

    def __repr__(self) -> str:
        return f"VectorStore(path={self._path!r}, vectors={len(self)}, dim={self._dim})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        with open(self._vecs_path, "rb") as f:
            header = f.read(_HEADER_SIZE)
            if len(header) < _HEADER_SIZE:
                raise ValueError("Corrupt .vecs: header too short.")
            dim, count = struct.unpack(_HEADER_FMT, header)
            if dim == 0:
                raise ValueError("Corrupt .vecs: dim=0 is invalid.")
            expected_size = _HEADER_SIZE + count * dim * 4
            f.seek(0, 2)
            actual_size = f.tell()
            if actual_size != expected_size:
                raise ValueError(
                    f"Corrupt .vecs: expected {expected_size} bytes, got {actual_size}."
                )
            f.seek(_HEADER_SIZE)
            vec_fmt = f"{dim}f"
            vec_size = struct.calcsize(vec_fmt)
            self._dim = dim
            self._vectors = [list(struct.unpack(vec_fmt, f.read(vec_size))) for _ in range(count)]

        with open(self._meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        if len(self._metadata) != count:
            raise ValueError(
                f"Corrupt store: .vecs has {count} vectors but .json has {len(self._metadata)} entries."
            )

        self._corpus = [_tokenize(str(m.get(self._text_field, ""))) for m in self._metadata]


def _matches(metadata: dict[str, Any], filter: dict[str, Any]) -> bool:
    return all(metadata.get(k) == v for k, v in filter.items())
