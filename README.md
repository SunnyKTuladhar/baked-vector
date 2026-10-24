# baked-vector

Zero-dependency vector store for small-scale embedding search. No numpy, no server, no setup — just a binary file and a JSON file baked into your project.

Designed for under 10,000 vectors. Bring your own embeddings.

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

## Providers

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

## License

MIT
