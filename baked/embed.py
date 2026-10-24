from __future__ import annotations

from typing import Any, Callable

from .store import VectorStore


Embedder = Callable[[str], list[float]]


class EmbeddingStore:
    """VectorStore wrapper that auto-embeds text via a provided embedder callable."""

    def __init__(self, path: str, embedder: Embedder):
        self._store = VectorStore(path)
        self._embed = embedder

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> int:
        vec = self._embed(text)
        meta = {"text": text, **(metadata or {})}
        return self._store.add(vec, meta)

    def search(
        self,
        query: str,
        top_k: int = 5,
        radius: float | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        vec = self._embed(query)
        return self._store.search(vec, top_k=top_k, radius=radius, filter=filter)

    def remove(self, id: int) -> None:
        self._store.remove(id)

    def save(self) -> None:
        self._store.save()

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"EmbeddingStore(store={self._store!r})"


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------

def sentence_transformer_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Returns an embedder using a sentence-transformers model (install baked[st])."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Install sentence-transformers: pip install baked-vector[st]")

    model = SentenceTransformer(model_name)

    def embed(text: str) -> list[float]:
        return model.encode(text, normalize_embeddings=False).tolist()

    return embed


def openai_embedder(model: str = "text-embedding-3-small", api_key: str | None = None) -> Embedder:
    """Returns an embedder using OpenAI's API (install baked[openai])."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: pip install baked-vector[openai]")

    client = OpenAI(api_key=api_key)

    def embed(text: str) -> list[float]:
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding

    return embed


def azure_openai_embedder(
    deployment: str,
    endpoint: str,
    api_key: str,
    api_version: str = "2024-02-01",
) -> Embedder:
    """Returns an embedder using Azure OpenAI (install baked[openai])."""
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise ImportError("Install openai: pip install baked-vector[openai]")

    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

    def embed(text: str) -> list[float]:
        response = client.embeddings.create(input=text, model=deployment)
        return response.data[0].embedding

    return embed


def bedrock_embedder(
    model_id: str = "amazon.titan-embed-text-v2:0",
    region_name: str = "us-east-1",
) -> Embedder:
    """Returns an embedder using AWS Bedrock (install baked[bedrock])."""
    try:
        import boto3
    except ImportError:
        raise ImportError("Install boto3: pip install baked-vector[bedrock]")

    import json

    client = boto3.client("bedrock-runtime", region_name=region_name)

    def embed(text: str) -> list[float]:
        body = json.dumps({"inputText": text})
        response = client.invoke_model(modelId=model_id, body=body)
        result = json.loads(response["body"].read())
        return result["embedding"]

    return embed


def google_embedder(
    model: str = "models/text-embedding-004",
    api_key: str | None = None,
) -> Embedder:
    """Returns an embedder using Google Generative AI (install baked[google])."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Install google-generativeai: pip install baked-vector[google]")

    if api_key:
        genai.configure(api_key=api_key)

    def embed(text: str) -> list[float]:
        result = genai.embed_content(model=model, content=text)
        return result["embedding"]

    return embed
