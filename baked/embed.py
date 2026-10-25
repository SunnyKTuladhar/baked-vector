from __future__ import annotations

from typing import Any, Callable

from .store import VectorStore


Embedder = Callable[[str], list[float]]


class EmbeddingStore:
    """VectorStore wrapper that auto-embeds text via a provided embedder callable."""

    def __init__(self, path: str, embedder: Embedder, text_field: str = "text"):
        self._store = VectorStore(path, text_field=text_field)
        self._embed = embedder
        self._text_field = text_field

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> int:
        vec = self._embed(text)
        # text_field always set by us; user metadata cannot silently override it
        meta = {**(metadata or {}), self._text_field: text}
        return self._store.add(vec, meta)

    def add_many(self, texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> list[int]:
        """Embed and add multiple texts. Returns list of assigned ids."""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(texts) != len(metadatas):
            raise ValueError(
                f"Length mismatch: {len(texts)} texts but {len(metadatas)} metadatas."
            )
        return [self.add(text, meta) for text, meta in zip(texts, metadatas)]

    def search(
        self,
        query: str,
        top_k: int = 5,
        radius: float | None = None,
        filter: dict[str, Any] | None = None,
        text_query: str | None = None,
        alpha: float = 0.5,
    ) -> list[tuple[dict[str, Any], float]]:
        vec = self._embed(query)
        return self._store.search(
            vec, top_k=top_k, radius=radius, filter=filter,
            text_query=text_query, alpha=alpha,
        )

    def remove(self, id: int) -> None:
        self._store.remove(id)

    def update(
        self,
        id: int,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Re-embed and/or update metadata for an existing entry."""
        if text is None and metadata is None:
            raise ValueError("At least one of text or metadata must be provided.")
        if text is not None:
            vec = self._embed(text)
            merged_meta = {**(metadata or {}), self._text_field: text}
            self._store.update(id, embedding=vec, metadata=merged_meta)
        else:
            # Metadata-only update: preserve existing text field
            existing = dict(self._store._metadata[id])
            merged = {**existing, **metadata}
            self._store.update(id, metadata=merged)

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
