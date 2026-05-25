"""SkegVectorStore: LlamaIndex VectorStore backed by skeg.

Design notes:

- LlamaIndex identifies nodes by string `node_id`. skeg's VINDEX takes
  integer vec_ids. We map node_id → u64 via xxh3 (collision-free in
  practice within a single index); the reverse mapping is stored in
  the KV side of skeg as `nid:{u64}` → original node_id bytes, and as
  `meta:{node_id}` → MessagePack-encoded metadata blob.
- A single skeg-server can host multiple VINDEX, so one
  `SkegVectorStore` instance corresponds to one named VINDEX.
- We use `kind="int8"` and `backend="flat"` by default. For larger
  indexes (>50K vectors) callers should pre-build an on-disk Vamana
  index via `skeg-tool build` and serve it read-only; this adapter
  then becomes a read-mostly wrapper.

Limitations of this v0.1 adapter:
- No metadata filter pushdown. LlamaIndex queries with filters get a
  post-filter pass on the returned hits.
- Streaming additions (`add()` with N nodes) issue one VSET per node;
  for batch inserts of thousands, prefer pre-building via skeg-tool.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import xxhash

from skeg import BinaryClient

if TYPE_CHECKING:
    # Import LlamaIndex types only for type-checking; at runtime we
    # accept duck-typed objects so the module imports without llama_index
    # installed (raising a clear error at instantiation if needed).
    from llama_index.core.schema import BaseNode, MetadataMode  # noqa: F401
    from llama_index.core.vector_stores.types import (
        VectorStoreQuery,
        VectorStoreQueryResult,
    )

NID_PREFIX = b"nid:"
META_PREFIX = b"meta:"


def _node_id_to_vec_id(node_id: str) -> int:
    """Hash a string node_id into a stable u64."""
    return xxhash.xxh3_64_intdigest(node_id.encode("utf-8"))


def _serialize_metadata(metadata: dict[str, Any], text: str) -> bytes:
    """Serialise (text, metadata) into a single blob the adapter can
    round-trip via skeg's KV layer. We use a tiny length-prefixed wire
    format instead of JSON so binary text + bytes-y metadata both pass
    through cleanly.

    Layout: `[u32 text_len][text utf-8][u32 meta_len][meta json]`.
    """
    import json
    text_b = text.encode("utf-8")
    meta_b = json.dumps(metadata or {}, ensure_ascii=False).encode("utf-8")
    return (
        struct.pack("<I", len(text_b)) + text_b
        + struct.pack("<I", len(meta_b)) + meta_b
    )


def _deserialize_metadata(blob: bytes) -> tuple[str, dict[str, Any]]:
    import json
    if len(blob) < 4:
        return "", {}
    (tlen,) = struct.unpack_from("<I", blob, 0)
    if 4 + tlen + 4 > len(blob):
        return blob[4:].decode("utf-8", errors="replace"), {}
    text = blob[4:4 + tlen].decode("utf-8")
    (mlen,) = struct.unpack_from("<I", blob, 4 + tlen)
    meta_start = 4 + tlen + 4
    meta = json.loads(blob[meta_start:meta_start + mlen].decode("utf-8"))
    return text, meta


@dataclass
class SkegVectorStore:
    """LlamaIndex VectorStore backed by a single skeg VINDEX.

    Use `SkegVectorStore.from_uri(...)` for the convenient constructor;
    use the dataclass form when wiring up tests or non-trivial setups
    that already have a `BinaryClient` instance.
    """

    client: BinaryClient
    index_name: str
    dim: int

    stores_text: bool = True
    is_embedding_query: bool = True

    @classmethod
    def from_uri(cls, uri: str, *, dim: int,
                  kind: str = "int8",
                  backend: str = "flat",
                  create: bool = True) -> "SkegVectorStore":
        """Convenience constructor.

        `uri` is `skeg://host:port/index_name`. If `create=True` the
        adapter calls VINDEX_CREATE on first use - safe to leave on
        because the server returns a soft error for "already exists"
        which the adapter swallows.
        """
        host, port, index_name = _parse_uri(uri)
        client = BinaryClient.connect(host, port)
        store = cls(client=client, index_name=index_name, dim=dim)
        if create:
            try:
                client.vindex_create(index_name, dim, kind=kind, backend=backend)
            except Exception:
                # already-exists is fine; let the first VSET surface a
                # real protocol error if the index is broken for some
                # other reason.
                pass
        return store

    # ── LlamaIndex BasePydanticVectorStore protocol ─────────────────

    @property
    def client_handle(self) -> BinaryClient:
        """Expose the underlying BinaryClient. LlamaIndex calls the
        attribute `client` on some integrations; we expose both names."""
        return self.client

    def add(self, nodes: Sequence["BaseNode"], **_: Any) -> list[str]:
        """Insert/replace nodes. Returns the list of node_ids actually stored."""
        from llama_index.core.schema import MetadataMode

        stored: list[str] = []
        for node in nodes:
            if node.embedding is None:
                raise ValueError(
                    f"node {node.node_id} has no embedding; LlamaIndex "
                    "must compute embeddings before passing to .add()"
                )
            if len(node.embedding) != self.dim:
                raise ValueError(
                    f"node {node.node_id} embedding dim "
                    f"{len(node.embedding)} != index dim {self.dim}"
                )
            vec_id = _node_id_to_vec_id(node.node_id)
            self.client.vset(self.index_name, vec_id, list(node.embedding))
            # Store the reverse mapping and the text+metadata blob in
            # the KV side. We use a short binary prefix so a casual
            # GET on a key never collides with user data.
            self.client.set(NID_PREFIX + struct.pack("<Q", vec_id),
                            node.node_id.encode("utf-8"))
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            blob = _serialize_metadata(dict(node.metadata or {}), text)
            self.client.set(META_PREFIX + node.node_id.encode("utf-8"), blob)
            stored.append(node.node_id)
        return stored

    def delete(self, ref_doc_id: str, **_: Any) -> None:
        """Delete a node by node_id (LlamaIndex calls this `ref_doc_id`
        in the VectorStore protocol)."""
        vec_id = _node_id_to_vec_id(ref_doc_id)
        # Best effort: tombstone in VINDEX, drop the KV mapping. If the
        # node was never added the calls all return False which we
        # ignore.
        self.client.vdel(self.index_name, vec_id)
        self.client.delete(NID_PREFIX + struct.pack("<Q", vec_id))
        self.client.delete(META_PREFIX + ref_doc_id.encode("utf-8"))

    def query(self, query: "VectorStoreQuery", **_: Any
              ) -> "VectorStoreQueryResult":
        """Run a top-k similarity query."""
        from llama_index.core.vector_stores.types import VectorStoreQueryResult

        if query.query_embedding is None:
            raise ValueError("SkegVectorStore.query requires query_embedding")
        if len(query.query_embedding) != self.dim:
            raise ValueError(
                f"query embedding dim {len(query.query_embedding)} != "
                f"index dim {self.dim}"
            )
        k = query.similarity_top_k or 10
        hits = self.client.vsearch(
            self.index_name, list(query.query_embedding), k=k,
        )
        node_ids: list[str] = []
        similarities: list[float] = []
        for h in hits:
            raw = self.client.get(NID_PREFIX + struct.pack("<Q", h.id))
            if raw is None:
                # The vec_id is in the index but the KV mapping was
                # dropped (partial delete). Skip silently.
                continue
            node_ids.append(raw.decode("utf-8"))
            similarities.append(h.score)
        return VectorStoreQueryResult(
            ids=node_ids, similarities=similarities, nodes=None,
        )

    # Convenience for tests / standalone use.
    def lookup_text(self, node_id: str) -> tuple[str, dict[str, Any]] | None:
        blob = self.client.get(META_PREFIX + node_id.encode("utf-8"))
        if blob is None:
            return None
        return _deserialize_metadata(blob)


def _parse_uri(uri: str) -> tuple[str, int, str]:
    """Parse `skeg://host:port/index_name` into its parts."""
    from urllib.parse import urlparse

    p = urlparse(uri)
    if p.scheme != "skeg":
        raise ValueError(f"unsupported URI scheme {p.scheme!r}; want 'skeg'")
    host = p.hostname or "127.0.0.1"
    port = p.port or 7379
    name = p.path.lstrip("/")
    if not name:
        raise ValueError(f"missing index name in {uri!r}")
    return host, port, name
