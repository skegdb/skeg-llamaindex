"""SkegVectorStore integration tests.

The tests need llama-index-core installed; if it is not, the file
skips automatically. We don't list llama-index-core in install_requires
of the adapter so users can still install the module for type checking.
"""
from __future__ import annotations

import random

import pytest

llama_index = pytest.importorskip("llama_index.core")

import numpy as np
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

from skeg_llamaindex import SkegVectorStore


def _vec(dim: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v] if norm > 0 else v


@pytest.fixture
def store(binary_server: dict, request: pytest.FixtureRequest
          ) -> SkegVectorStore:
    # Per-test VINDEX name avoids cross-test interference.
    index = f"notes-{request.node.name}"
    uri = f"skeg://{binary_server['host']}:{binary_server['port']}/{index}"
    store = SkegVectorStore.from_uri(uri, dim=8, kind="f32", backend="flat")
    yield store
    try:
        store.client.vindex_drop(index)
    except Exception:
        pass
    store.client.close()


def test_add_then_query_finds_node(store: SkegVectorStore) -> None:
    nodes = []
    for i in range(5):
        n = TextNode(text=f"text-{i}", embedding=_vec(8, seed=i + 100))
        nodes.append(n)
    stored = store.add(nodes)
    assert len(stored) == 5

    # Query with the embedding of node 0; node 0 must be the top hit.
    q = VectorStoreQuery(
        query_embedding=nodes[0].embedding, similarity_top_k=3,
    )
    result = store.query(q)
    assert nodes[0].node_id in result.ids
    assert result.ids[0] == nodes[0].node_id  # closest first


def test_add_rejects_dim_mismatch(store: SkegVectorStore) -> None:
    n = TextNode(text="wrong-dim", embedding=[0.0] * 16)  # store dim is 8
    with pytest.raises(ValueError, match="dim"):
        store.add([n])


def test_lookup_text_returns_text_and_metadata(store: SkegVectorStore) -> None:
    n = TextNode(
        text="hello world", embedding=_vec(8, seed=42),
        metadata={"source": "test"},
    )
    store.add([n])
    out = store.lookup_text(n.node_id)
    assert out is not None
    text, meta = out
    assert text == "hello world"
    assert meta.get("source") == "test"


def test_delete_removes_node(store: SkegVectorStore) -> None:
    n = TextNode(text="to-delete", embedding=_vec(8, seed=99))
    store.add([n])
    store.delete(n.node_id)
    # After delete, lookup_text should return None and a query for the
    # node's exact embedding should not list it.
    assert store.lookup_text(n.node_id) is None
    q = VectorStoreQuery(query_embedding=n.embedding, similarity_top_k=1)
    result = store.query(q)
    assert n.node_id not in result.ids
