from .store import VectorStore
from .embed import (
    EmbeddingStore,
    sentence_transformer_embedder,
    openai_embedder,
    azure_openai_embedder,
    bedrock_embedder,
    google_embedder,
)
from .chunking import chunk_fixed, chunk_sentences, chunk_paragraphs, chunk_recursive

__all__ = [
    "VectorStore",
    "EmbeddingStore",
    "sentence_transformer_embedder",
    "openai_embedder",
    "azure_openai_embedder",
    "bedrock_embedder",
    "google_embedder",
    "chunk_fixed",
    "chunk_sentences",
    "chunk_paragraphs",
    "chunk_recursive",
]
from importlib.metadata import version as _v, PackageNotFoundError as _E
try:
    __version__ = _v("baked-vector")
except _E:
    __version__ = "unknown"
