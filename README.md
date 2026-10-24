# baked-vector

Zero-dependency vector store for small-scale embedding search. No numpy, no server, no setup — just a binary file and a JSON file baked into your project.

Designed for under 10,000 vectors. Bring your own embeddings.

## Install

```bash
pip install baked-vector
```

## Usage

```python
import baked

store = baked.VectorStore("mydata")

# add vectors with arbitrary metadata
store.add(embedding, {"source": "doc1.pdf", "page": 3, "text": "..."})
store.save()

# reload from disk automatically
store = baked.VectorStore("mydata")

# search
results = store.search(query_embedding, top_k=5)
# [(metadata_dict, score), ...]

# filter by metadata
results = store.search(query_embedding, top_k=5, filter={"source": "doc1.pdf"})

# range search — all results with score >= threshold
results = store.search(query_embedding, radius=0.85)

# batch search
results = store.search_many([vec1, vec2], top_k=5)
```

## API

### `VectorStore(path)`

Loads from `<path>.vecs` and `<path>.json` if they exist, otherwise starts empty.

| Method | Description |
|--------|-------------|
| `add(embedding, metadata)` | Add a vector. Returns its id. |
| `remove(id)` | Soft-delete. Takes effect on next `save()`. |
| `save()` | Flush to disk, compacting deleted entries. |
| `search(query, top_k, radius, filter)` | Find similar vectors. |
| `search_many(queries, top_k, radius, filter)` | Batch search. |
| `len(store)` | Number of active vectors. |

### Search parameters

- `top_k` — return top N results (default: 5)
- `radius` — return all results with score ≥ threshold (overrides `top_k`)
- `filter` — dict of key-value pairs that must match the metadata exactly

Scores are cosine similarity in range [-1, 1]. Higher = more similar.

## Storage format

Two files:

- `<path>.vecs` — binary packed floats. 8-byte header (dim, count), then N × dim × 4 bytes.
- `<path>.json` — JSON array of metadata dicts, one per vector.

Vectors are L2-normalized at insert time. IDs are positional and shift after `save()` following a `remove()`.

## Performance

Pure Python, brute-force scan. Rough query times:

| Vectors | 384-dim | 768-dim | 1536-dim |
|---------|---------|---------|----------|
| 1k | <0.1s | <0.1s | ~0.1s |
| 5k | ~0.1s | ~0.3s | ~0.5s |
| 10k | ~0.3s | ~0.6s | ~1–2s |

## License

MIT
