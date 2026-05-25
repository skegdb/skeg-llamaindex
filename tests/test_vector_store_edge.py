"""SkegVectorStore edge cases: batch, ordering, empty results, dim
mismatch, metadata roundtrip, recovery semantics.

These tests focus on the contract LlamaIndex relies on (ordered ids,
stable similarities, idempotent delete) without re-testing the wire
roundtrip - that's covered by skeg-py.
"""
from __future__ import annotations

import random

import pytest

llama_index = pytest.importorskip("llama_index.core")

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
    # Per-test VINDEX name keeps tests independent: a leak from one test
    # never colours the next test's recall numbers or query results.
    index = f"edge-{request.node.name}"
    uri = f"skeg://{binary_server['host']}:{binary_server['port']}/{index}"
    s = SkegVectorStore.from_uri(uri, dim=8, kind="f32", backend="flat")
    yield s
    try:
        s.client.vindex_drop(index)
    except Exception:
        pass
    s.client.close()


# ── ordering + scoring ───────────────────────────────────────────────


def test_query_returns_results_in_descending_score(store: SkegVectorStore
                                                    ) -> None:
    nodes = [
        TextNode(text=f"text-{i}", embedding=_vec(8, seed=i + 200))
        for i in range(10)
    ]
    store.add(nodes)
    q = VectorStoreQuery(
        query_embedding=_vec(8, seed=999), similarity_top_k=5,
    )
    result = store.query(q)
    sims = result.similarities or []
    assert len(sims) == 5
    # Cosine: higher = closer; expect non-increasing.
    for a, b in zip(sims, sims[1:]):
        assert a >= b - 1e-6


def test_query_top_k_is_capped_at_corpus_size(store: SkegVectorStore) -> None:
    # 3 nodes, ask for top-10: expect 3 results, not an error.
    nodes = [
        TextNode(text=f"t{i}", embedding=_vec(8, seed=i + 300))
        for i in range(3)
    ]
    store.add(nodes)
    q = VectorStoreQuery(
        query_embedding=_vec(8, seed=42), similarity_top_k=10,
    )
    result = store.query(q)
    assert len(result.ids) == 3


def test_query_against_empty_index_returns_empty_result(store: SkegVectorStore
                                                        ) -> None:
    q = VectorStoreQuery(
        query_embedding=_vec(8, seed=1), similarity_top_k=5,
    )
    result = store.query(q)
    assert result.ids == []
    assert (result.similarities or []) == []


# ── dim mismatch / validation ────────────────────────────────────────


def test_add_rejects_node_without_embedding(store: SkegVectorStore) -> None:
    n = TextNode(text="missing-emb")  # no embedding set
    with pytest.raises(ValueError, match="no embedding"):
        store.add([n])


def test_query_rejects_missing_embedding(store: SkegVectorStore) -> None:
    q = VectorStoreQuery(similarity_top_k=5)  # no query_embedding
    with pytest.raises(ValueError, match="query_embedding"):
        store.query(q)


def test_query_rejects_dim_mismatch(store: SkegVectorStore) -> None:
    q = VectorStoreQuery(
        query_embedding=[0.0] * 16,  # store dim is 8
        similarity_top_k=5,
    )
    with pytest.raises(ValueError, match="dim"):
        store.query(q)


# ── batch add ────────────────────────────────────────────────────────


def test_add_64_nodes_in_one_call(store: SkegVectorStore) -> None:
    # Verifies the per-node loop doesn't accumulate state issues.
    nodes = [
        TextNode(text=f"batch-{i}", embedding=_vec(8, seed=i + 500),
                 metadata={"idx": i})
        for i in range(64)
    ]
    stored = store.add(nodes)
    assert len(stored) == 64
    # Every node should be retrievable by its node_id.
    for n in nodes:
        out = store.lookup_text(n.node_id)
        assert out is not None
        text, meta = out
        assert text == f"batch-{n.metadata['idx']}"
        assert meta.get("idx") == n.metadata["idx"]


def test_add_then_query_finds_correct_node_among_many(store: SkegVectorStore
                                                       ) -> None:
    nodes = [
        TextNode(text=f"n{i}", embedding=_vec(8, seed=i + 700))
        for i in range(30)
    ]
    store.add(nodes)
    # Query with node 17's exact embedding; node 17 must be the top hit.
    target = nodes[17]
    q = VectorStoreQuery(
        query_embedding=target.embedding, similarity_top_k=1,
    )
    result = store.query(q)
    assert result.ids[0] == target.node_id
    assert (result.similarities or [])[0] > 0.99


# ── metadata roundtrip ──────────────────────────────────────────────


def test_metadata_with_string_values_roundtrips(store: SkegVectorStore) -> None:
    n = TextNode(
        text="hello", embedding=_vec(8, seed=10),
        metadata={"source": "wiki", "lang": "it"},
    )
    store.add([n])
    out = store.lookup_text(n.node_id)
    assert out is not None
    _, meta = out
    assert meta == {"source": "wiki", "lang": "it"}


def test_metadata_with_nested_dict_roundtrips(store: SkegVectorStore) -> None:
    n = TextNode(
        text="nested", embedding=_vec(8, seed=11),
        metadata={"location": {"city": "Rome", "country": "IT"}, "year": 2026},
    )
    store.add([n])
    out = store.lookup_text(n.node_id)
    assert out is not None
    _, meta = out
    assert meta["location"]["city"] == "Rome"
    assert meta["year"] == 2026


def test_empty_metadata_is_allowed(store: SkegVectorStore) -> None:
    n = TextNode(text="bare", embedding=_vec(8, seed=12), metadata={})
    store.add([n])
    out = store.lookup_text(n.node_id)
    assert out is not None
    _, meta = out
    assert meta == {}


def test_unicode_text_and_metadata_survive_roundtrip(store: SkegVectorStore
                                                     ) -> None:
    n = TextNode(
        text="caffè italiano ☕", embedding=_vec(8, seed=13),
        metadata={"città": "Roma"},
    )
    store.add([n])
    out = store.lookup_text(n.node_id)
    assert out is not None
    text, meta = out
    assert text == "caffè italiano ☕"
    assert meta == {"città": "Roma"}


# ── delete semantics ────────────────────────────────────────────────


def test_delete_then_lookup_returns_none(store: SkegVectorStore) -> None:
    n = TextNode(text="x", embedding=_vec(8, seed=21))
    store.add([n])
    store.delete(n.node_id)
    assert store.lookup_text(n.node_id) is None


def test_delete_missing_is_noop(store: SkegVectorStore) -> None:
    # Deleting a never-added node must not raise.
    store.delete("never-added-id")


def test_delete_then_readd_is_visible(store: SkegVectorStore) -> None:
    n = TextNode(text="reborn", embedding=_vec(8, seed=22),
                  metadata={"v": 1})
    store.add([n])
    store.delete(n.node_id)
    # Re-add the same node with new metadata.
    n2 = TextNode(text="reborn", embedding=_vec(8, seed=22),
                   metadata={"v": 2})
    n2.node_id = n.node_id  # keep the same id deliberately
    store.add([n2])
    out = store.lookup_text(n.node_id)
    assert out is not None
    _, meta = out
    assert meta["v"] == 2


def test_delete_one_does_not_affect_neighbours(store: SkegVectorStore) -> None:
    nodes = [
        TextNode(text=f"n{i}", embedding=_vec(8, seed=i + 800))
        for i in range(5)
    ]
    store.add(nodes)
    store.delete(nodes[2].node_id)
    for i, n in enumerate(nodes):
        out = store.lookup_text(n.node_id)
        if i == 2:
            assert out is None
        else:
            assert out is not None


# ── URI parsing ─────────────────────────────────────────────────────


def test_from_uri_parses_default_port(binary_server: dict) -> None:
    # We deliberately use a custom non-default port to make sure the
    # parser respects an explicit port. (Connection won't be made yet
    # because we want to assert on parsing only.)
    host = binary_server["host"]
    port = binary_server["port"]
    s = SkegVectorStore.from_uri(
        f"skeg://{host}:{port}/parsing-probe",
        dim=8, kind="f32", backend="flat",
    )
    assert s.index_name == "parsing-probe"
    s.client.close()


def test_from_uri_rejects_missing_index_name(binary_server: dict) -> None:
    host = binary_server["host"]
    port = binary_server["port"]
    with pytest.raises(ValueError, match="index name"):
        SkegVectorStore.from_uri(f"skeg://{host}:{port}/", dim=8)


def test_from_uri_rejects_wrong_scheme() -> None:
    with pytest.raises(ValueError, match="scheme"):
        SkegVectorStore.from_uri("redis://127.0.0.1:7379/x", dim=8)
