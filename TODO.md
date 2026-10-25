# TODO

## Release

### Publish to PyPI ✓
- ~~Create account at pypi.org + enable 2FA~~
- ~~Verify name is free: `pypi.org/project/baked-vector`~~
- ~~Publish: `uv build && uv publish`~~
- `0.1.0` live at https://pypi.org/project/baked-vector/
- OIDC trusted publisher configured (no API tokens)
- Push to `main` → auto-publish to TestPyPI
- Push tag `v*` → auto-publish to PyPI

---

## Feature backlog

Feature backlog for baked-vector, ordered by priority for RAG / chatbot use cases.

---

## Priority 1 — Core RAG gaps

### Hybrid Search
**Keyword (BM25-style) + semantic search, merged via Reciprocal Rank Fusion (RRF).**
- Pure Python BM25 is ~30 lines, zero deps
- Biggest single retrieval quality improvement for RAG — semantic alone misses exact keyword matches
- `store.search(query, top_k=5, alpha=0.5)` where alpha balances keyword vs semantic weight

### Namespaces / Collections
**Multiple logical stores in one file path.**
- `VectorStore("db", namespace="users")` — isolates tenant data without separate files
- Critical for multi-tenant chatbots and multi-domain search
- Stored as separate binary segments or separate files with a shared manifest

### Semantic Cache
**Cache LLM responses by query similarity — return cached answer if a near-identical query was seen.**
- Cuts LLM API costs dramatically for repeated or paraphrased questions
- Natural fit: the library already does similarity search, semantic cache is just `search(query, radius=0.95)` + a stored response field
- `SemanticCache("cache", embedder=embed, threshold=0.95)`

---

## Priority 2 — Retrieval quality

### MMR Search (Maximal Marginal Relevance)
**Return diverse results instead of N near-duplicate top results.**
- One extra parameter: `store.search(query, top_k=5, diversity=0.5)`
- Avoids returning five chunks from the same paragraph — common pain point in RAG
- Pure Python, ~20 lines on top of existing search

### Re-ranking
**Retrieve top-20 by vector search, re-rank with a cross-encoder or LLM judge, return top-5.**
- Higher precision than vector search alone at minimal extra cost
- `reranker` callable on `EmbeddingStore` — keeps it pluggable (cross-encoder, Cohere rerank API, LLM judge)
- Follows existing adapter pattern from `embed.py`

### Advanced Metadata Filtering
**Range queries, `$in`, `$not`, `$gte`, `$lte` — not just equality.**
- `filter={"year": {"$gte": 2020}, "tag": {"$in": ["ai", "ml"]}}`
- Current equality-only filter is too limiting for real document collections
- Implemented as a filter DSL parsed before scoring — no dep needed

### Deduplication on Insert
**Skip or warn if new vector is within threshold of an existing one.**
- Prevents bloating the store with near-identical chunks from overlapping ingestion
- `store.add(vec, metadata, dedup_threshold=0.98)`

---

## Priority 3 — Developer experience

### Document Loaders
**`baked.loaders` — load and chunk files ready for `add_many()`.**
- `from_txt(path)`, `from_markdown(path)`, `from_url(url)` — stdlib only
- `from_pdf(path)` — optional dep (`pypdf`)
- Returns `(chunks, metadatas)` tuples directly usable with `EmbeddingStore.add_many()`

### CLI
**`baked` command-line tool for ingestion and search without writing code.**
- `baked add docs/ --store mydb --embedder openai` — bulk ingest a directory
- `baked search "query" --store mydb` — one-shot retrieval
- `baked stats mydb` — show vector count, dim, disk size
- Useful for debugging and one-off ingestion pipelines

### `stats()` / Introspection
**Surface store internals in one call.**
- `store.stats()` → `{"count": 1000, "dim": 1536, "deleted": 3, "size_bytes": 9437184}`
- Developers constantly want to know what's in their store
- Trivially derived from existing state, zero new logic

### Async Support
**`AsyncEmbeddingStore` wrapping async embed callables.**
- File I/O is sync and fast — async matters for the embedding API calls, not storage
- `async def search(query)`, `async def add(text, metadata)`
- Needed for FastAPI and async chatbot backends

### LangChain / LlamaIndex Adapter
**Implement their `VectorStore` interface so baked-vector is a drop-in.**
- Opens adoption to the large existing LangChain/LlamaIndex user base
- No new logic — thin adapter class delegating to `EmbeddingStore`
- Optional extra: `baked-vector[langchain]`

---

## Priority 4 — Storage and performance

### int8 Quantization
**Store vectors as int8 instead of float32 — 4x smaller file.**
- 10k × 1536-dim store: 60 MB → 15 MB
- Reconstruct to float at search time with a linear scale factor
- `VectorStore("db", quantize=True)`

### Append-Only Writes
**`add()` appends to file without rewriting everything.**
- Current `save()` rewrites the full binary file — slow for frequent inserts at 10k+ vectors
- Append-only log + periodic compaction on `save()`

### Export / Import
**Serialize and deserialize the full store.**
- `store.export("dump.json")` — human-readable, portable
- `VectorStore.from_json("dump.json")` — restore from export
- Needed for migration between machines, backup, and debugging

### Point-in-Time Snapshots
**`store.snapshot("backup.vecs")` — cheap copy of current state.**
- Useful before bulk updates or destructive operations
- Simple file copy of the two store files

### TTL / Expiry
**Metadata field `_expires_at` — auto-skip expired vectors in search.**
- Useful for caching (expire cached LLM responses) and time-sensitive content
- `store.add(vec, {"_expires_at": timestamp, ...})`
- Check `time.time()` at search time, no background process needed
