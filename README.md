# baked-vector

Zero-dependency vector store for small-scale embedding search. No numpy, no server, no setup — just a binary file and a JSON file baked into your project.

Designed for under 10,000 vectors. Bring your own embeddings.

## Why baked-vector?

| | baked-vector | ChromaDB | FAISS |
|---|---|---|---|
| Install | `pip install baked-vector` | `pip install chromadb` + system deps | `pip install faiss-cpu` + numpy |
| Dependencies | **0** | 20+ | numpy + native libs |
| Setup | None | Server or persistent client | Index build |
| Storage | Two plain files | SQLite + proprietary files | Binary index file |
| Hybrid search | ✅ BM25 + semantic | ❌ | ❌ |
| Best for | Small RAG, scripts, local tools | Mid-scale apps | High-scale ANN search |

## Install

```bash
pip install baked-vector
# or
uv add baked-vector
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

### `VectorStore(path, text_field="text")`

Loads from `<path>.vecs` and `<path>.json` if they exist, otherwise starts empty. `text_field` names the metadata key used for BM25 keyword indexing in hybrid search.

| Method | Description |
|--------|-------------|
| `add(embedding, metadata)` | Add a vector. Returns its id. |
| `remove(id)` | Soft-delete. Takes effect on next `save()`. |
| `update(id, embedding, metadata)` | Update vector and/or metadata in place. |
| `save()` | Flush to disk, compacting deleted entries. |
| `search(query, top_k, radius, filter, text_query, alpha)` | Find similar vectors. |
| `search_many(queries, top_k, radius, filter, text_query, alpha)` | Batch search. |
| `len(store)` | Number of active vectors. |

### Search parameters

- `top_k` — return top N results (default: 5)
- `radius` — return all results with score ≥ threshold (overrides `top_k`)
- `filter` — dict of key-value pairs that must match the metadata exactly
- `text_query` — keyword query string for hybrid search (BM25 + semantic)
- `alpha` — blend weight: `1.0` = pure semantic, `0.0` = pure BM25, `0.5` = balanced (default)

Scores are cosine similarity in range [-1, 1] for pure semantic search. Hybrid scores are min-max normalized to [0, 1] across the result set — scores are relative, not absolute. The top result always scores 1.0 in hybrid mode.

### Hybrid search

Combine keyword and semantic search with a single call:

```python
store = baked.VectorStore("mydata", text_field="text")
store.add(embedding, {"text": "python machine learning tutorial"})
store.add(embedding2, {"text": "python snake reptile habitat"})

# semantic only
results = store.search(query_embedding, top_k=5)

# hybrid — keyword + semantic, balanced
results = store.search(query_embedding, text_query="machine learning", top_k=5, alpha=0.5)

# pure keyword (BM25)
results = store.search(query_embedding, text_query="machine learning", top_k=5, alpha=0.0)
```

## Storage format

Two files:

- `<path>.vecs` — binary packed floats. 8-byte header (dim, count), then N × dim × 4 bytes.
- `<path>.json` — JSON array of metadata dicts, one per vector.

Vectors are L2-normalized at insert time. IDs are positional and shift after `save()` following a `remove()`.

## Performance

Benchmarked on 109 Paul Graham essays split into **3,421 chunks** (~556 chars each), embedded with `all-MiniLM-L6-v2` (384-dim), on a consumer laptop CPU. 327 eval pairs, Recall@k and MRR measured across all pairs.

### Retrieval quality

| Search mode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@5 |
|-------------|----------|----------|----------|-----------|-------|
| Pure semantic | 0.667 | 0.801 | 0.841 | 0.878 | 0.734 |
| Hybrid α=0.7 (lean semantic) | 0.795 | 0.905 | 0.927 | 0.948 | 0.848 |
| **Hybrid α=0.5 (balanced)** | **0.820** | **0.908** | **0.936** | **0.954** | **0.866** |
| Hybrid α=0.3 (lean BM25) | 0.810 | 0.902 | 0.920 | 0.948 | 0.857 |
| Pure BM25 / keyword | 0.765 | 0.872 | 0.884 | 0.927 | 0.815 |

Hybrid at α=0.5 beats pure semantic by **+9.5pp Recall@5** and pure BM25 by **+5.2pp**.

### Query latency (3,421 vectors, 384-dim, top_k=5)

| Search mode | avg | p50 | p95 |
|-------------|-----|-----|-----|
| Pure semantic (embed + scan) | 82ms | 82ms | 100ms |
| Hybrid (embed + scan + BM25) | 187ms | 188ms | 209ms |
| Pure BM25 (embed + BM25 scan) | 185ms | 186ms | 204ms |

The 82ms end-to-end for pure semantic includes the local `sentence-transformers` embed call (~65ms) plus the in-memory dot-product scan (~17ms). Hybrid adds a BM25 corpus scan — roughly 2× total latency. With a remote embedder (e.g. Bedrock, OpenAI), the scan cost is negligible.

## EmbeddingStore

`EmbeddingStore` wraps `VectorStore` and handles embedding automatically. Install an extra for your provider, pass the adapter, and work with text directly.

```python
import baked

embed = baked.openai_embedder()  # see adapters below
store = baked.EmbeddingStore("mydata", embedder=embed)

store.add("Wealth is not money...", {"title": "How to Make Wealth"})
store.save()

results = store.search("how startups create value", top_k=3)
# [({"text": "...", "title": "..."}, score), ...]
```

`EmbeddingStore` exposes the same `add`, `remove`, `save`, `search`, and `len` interface as `VectorStore`, but takes `str` instead of `list[float]`. The original text is stored in metadata under `"text"` automatically.

### Adapters

| Provider | Function | Install extra |
|----------|----------|---------------|
| OpenAI | `openai_embedder(model, api_key)` | `baked-vector[openai]` |
| Azure OpenAI | `azure_openai_embedder(deployment, endpoint, api_key, api_version)` | `baked-vector[openai]` |
| AWS Bedrock | `bedrock_embedder(model_id, region_name)` | `baked-vector[bedrock]` |
| Google | `google_embedder(model, api_key)` | `baked-vector[google]` |
| Local (sentence-transformers) | `sentence_transformer_embedder(model_name)` | `baked-vector[st]` |

```bash
pip install baked-vector[openai]
pip install baked-vector[bedrock]
pip install baked-vector[google]
pip install baked-vector[st]
```

Defaults: OpenAI → `text-embedding-3-small`, Bedrock → `amazon.titan-embed-text-v2:0`, Google → `models/text-embedding-004`, local → `all-MiniLM-L6-v2`.

You can also pass any `(str) -> list[float]` callable as `embedder` — no adapter required.

---

## Providers (manual)

baked-vector stores and searches vectors — bring embeddings from any provider.

### OpenAI

```python
from openai import OpenAI

client = OpenAI()

def embed(text: str) -> list[float]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding

store.add(embed("hello world"), {"text": "hello world"})
results = store.search(embed("hi there"), top_k=5)
```

### AWS Bedrock (Titan)

```python
import json
import boto3

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def embed(text: str) -> list[float]:
    resp = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text, "dimensions": 512}),
    )
    return json.loads(resp["body"].read())["embedding"]

store.add(embed("hello world"), {"text": "hello world"})
results = store.search(embed("hi there"), top_k=5)
```

### Azure OpenAI

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="YOUR_KEY",
    api_version="2024-02-01",
    azure_endpoint="https://YOUR_RESOURCE.openai.azure.com",
)

def embed(text: str) -> list[float]:
    resp = client.embeddings.create(model="YOUR_DEPLOYMENT_NAME", input=text)
    return resp.data[0].embedding

store.add(embed("hello world"), {"text": "hello world"})
results = store.search(embed("hi there"), top_k=5)
```

### Cohere

```python
import cohere

co = cohere.Client("YOUR_API_KEY")

def embed(text: str) -> list[float]:
    resp = co.embed(texts=[text], model="embed-english-v3.0", input_type="search_document")
    return resp.embeddings[0]

store.add(embed("hello world"), {"text": "hello world"})

# use input_type="search_query" for query embeddings
results = store.search(
    co.embed(texts=["hi there"], model="embed-english-v3.0", input_type="search_query").embeddings[0],
    top_k=5,
)
```

## Chunking

`baked` ships text chunking utilities for splitting documents before embedding. All are stdlib-only with no external dependencies.

```python
from baked import chunk_fixed, chunk_sentences, chunk_paragraphs, chunk_recursive

# Fixed character window — good for code, transcripts, uniform text
chunks = chunk_fixed(text, size=512, overlap=64)

# Split on sentence boundaries (.!?) — good for prose
chunks = chunk_sentences(text, max_size=512, overlap=1)

# Split on blank lines — good for markdown, structured docs
chunks = chunk_paragraphs(text, max_size=1024, overlap=0)

# LangChain-style recursive — tries \n\n → \n → ". " → hard cut
chunks = chunk_recursive(text, size=512, overlap=64)
```

All functions return `list[str]`. `max_size` / `size` is a soft limit for sentence and paragraph chunkers — a single item larger than the limit is kept as its own chunk.

## License

MIT
