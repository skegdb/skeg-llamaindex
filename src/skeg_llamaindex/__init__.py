"""LlamaIndex VectorStore adapter for skeg.

Subclasses `llama_index.core.vector_stores.types.BasePydanticVectorStore`
so users can drop skeg in wherever they currently have FAISS, Chroma,
or Qdrant configured.

Quick start:

    from skeg_llamaindex import SkegVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext

    store = SkegVectorStore.from_uri(
        "skeg://127.0.0.1:7379/notes",
        dim=1024, kind="int8",
    )
    storage_ctx = StorageContext.from_defaults(vector_store=store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_ctx)

Requires `llama-index-core >= 0.10`.
"""
from __future__ import annotations

from .vector_store import SkegVectorStore

__all__ = ["SkegVectorStore"]
__version__ = "0.1.0"
