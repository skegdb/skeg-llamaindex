"""`SkegVectorStore`: LlamaIndex `VectorStore` backed by a single skeg
VINDEX, plus a small slice of the KV side for text + metadata.

Design

- LlamaIndex identifies nodes by string `node_id`. skeg's VINDEX stores
  vectors under integer `u64` ids. The adapter maps `node_id -> u64`
  via xxh3-64 (collision-free in practice within one index) and stores
  the reverse mapping in the KV side as `nid:<u64 le>`. Text + metadata
  live under `meta:<node_id>` in a tiny length-prefixed blob.
- Because `stores_text = True`, `query()` rehydrates `TextNode` objects
  for every hit so downstream callers do not need a separate docstore
  round-trip. The reverse lookup of `<u64> -> node_id` and the
  `meta:<node_id> -> blob` reads are issued through `mget` to keep the
  hot path at two TCP round-trips total instead of `1 + 2k`.

Operational notes

- One adapter instance corresponds to one VINDEX. Hosting multiple
  VINDEX in the same server is fine; instantiate the adapter once per
  index.
- The underlying `BinaryClient` is single-connection and not
  thread-safe. Do not share one `SkegVectorStore` across threads or
  asyncio tasks; build one per worker.
- `add()` is not atomic: each node issues `vset` then two `set` calls.
  A network failure halfway through can leave the vector present in
  VINDEX without the KV reverse mapping, which `query()` then silently
  skips. For bulk inserts of large corpora prefer building the index
  offline with `skeg-cli build` and serving it read-only.

What this adapter does not do

- Metadata filter pushdown. `VectorStoreQuery.filters` is currently
  ignored; LlamaIndex post-filters the returned nodes itself.
- Sparse/hybrid search. skeg's index surface is dense-only.
- Async I/O. The adapter is synchronous; an async wrapper would need
  an async `BinaryClient` first.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any, Sequence
from urllib.parse import urlparse

import xxhash

from skeg import BinaryClient
from skeg.errors import ServerError

if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode, TextNode  # noqa: F401
    from llama_index.core.vector_stores.types import (
        VectorStoreQuery,
        VectorStoreQueryResult,
    )

# Key prefixes for the slice of skeg's KV side this adapter owns. The
# binary prefix is unlikely to collide with user keys (which are
# typically utf-8 strings).
NID_PREFIX = b"nid:"  # `nid:<u64 little-endian>` -> node_id utf-8 bytes
META_PREFIX = b"meta:"  # `meta:<node_id utf-8>` -> length-prefixed text+metadata blob

# Whitelisted VINDEX configuration values. The server validates these
# too; checking client-side gives a faster, clearer error.
_ALLOWED_KINDS = frozenset({"f32", "int8", "binary"})
_ALLOWED_BACKENDS = frozenset({"flat", "disk", "disk_vamana"})


def _node_id_to_vec_id(node_id: str) -> int:
    """Hash a string `node_id` into a stable u64."""
    return xxhash.xxh3_64_intdigest(node_id.encode("utf-8"))


def _nid_key(vec_id: int) -> bytes:
    return NID_PREFIX + struct.pack("<Q", vec_id)


def _meta_key(node_id: str) -> bytes:
    return META_PREFIX + node_id.encode("utf-8")


def _serialize_blob(text: str, metadata: dict[str, Any]) -> bytes:
    """Pack `(text, metadata)` into one blob round-trippable through KV.

    Layout: `[u32 text_len][text utf-8][u32 meta_len][meta json utf-8]`.
    A tiny length-prefixed wire avoids JSON-escaping the text body and
    keeps the metadata field self-delimiting.
    """
    text_b = text.encode("utf-8")
    meta_b = json.dumps(metadata or {}, ensure_ascii=False).encode("utf-8")
    return (
        struct.pack("<I", len(text_b))
        + text_b
        + struct.pack("<I", len(meta_b))
        + meta_b
    )


def _deserialize_blob(blob: bytes) -> tuple[str, dict[str, Any]]:
    """Inverse of `_serialize_blob`. A truncated or malformed blob
    yields `("", {})` rather than raising, so a partial-write race on
    the KV side cannot crash an entire query result."""
    if len(blob) < 4:
        return "", {}
    (tlen,) = struct.unpack_from("<I", blob, 0)
    if 4 + tlen + 4 > len(blob):
        return blob[4 : 4 + tlen].decode("utf-8", errors="replace"), {}
    text = blob[4 : 4 + tlen].decode("utf-8", errors="replace")
    (mlen,) = struct.unpack_from("<I", blob, 4 + tlen)
    meta_start = 4 + tlen + 4
    try:
        meta = json.loads(blob[meta_start : meta_start + mlen].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        meta = {}
    return text, meta


def _parse_uri(uri: str) -> tuple[str, int, str]:
    """Parse `skeg://host:port/index_name` into its parts."""
    p = urlparse(uri)
    if p.scheme != "skeg":
        raise ValueError(f"unsupported URI scheme {p.scheme!r}; want 'skeg'")
    host = p.hostname or "127.0.0.1"
    port = p.port or 7379
    name = p.path.lstrip("/")
    if not name:
        raise ValueError(f"missing index name in {uri!r}")
    return host, port, name


@dataclass
class SkegVectorStore:
    """LlamaIndex `VectorStore` backed by one skeg VINDEX.

    Construct via :meth:`from_uri` for the common case, or pass an
    existing :class:`skeg.BinaryClient` directly when the caller already
    manages connection lifetime (tests, custom retry layers).

    Attributes
    ----------
    client : BinaryClient
        Connected client. Owned by the store iff :meth:`from_uri` made
        it; in that case :meth:`close` will close it. When passed in
        explicitly the caller owns its lifetime.
    index_name : str
        VINDEX name on the server.
    dim : int
        Vector dimension. Must match every embedding passed to
        :meth:`add` and :meth:`query`.
    stores_text : bool, default True
        :meth:`query` populates `result.nodes` with `TextNode`
        objects.
    is_embedding_query : bool, default True
        LlamaIndex requires a precomputed query embedding (the adapter
        does not embed text itself).
    """

    client: BinaryClient
    index_name: str
    dim: int

    stores_text: bool = True
    is_embedding_query: bool = True

    _owns_client: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.dim, int) or self.dim <= 0:
            raise ValueError(f"dim must be a positive int, got {self.dim!r}")
        if not self.index_name:
            raise ValueError("index_name must not be empty")

    # ── Construction ─────────────────────────────────────────────────

    @classmethod
    def from_uri(
        cls,
        uri: str,
        *,
        dim: int,
        kind: str = "int8",
        backend: str = "flat",
        create: bool = True,
    ) -> "SkegVectorStore":
        """Open a `skeg://host:port/index_name` URI and return a store.

        If `create=True` (the default) the adapter calls
        `VINDEX.CREATE` and treats an "already exists" error from the
        server as success. Other `ServerError`s propagate so a misuse
        (e.g. wrong dim against an existing index) surfaces at setup
        time rather than at the first add().
        """
        if kind not in _ALLOWED_KINDS:
            raise ValueError(
                f"kind={kind!r} not in {sorted(_ALLOWED_KINDS)}"
            )
        if backend not in _ALLOWED_BACKENDS:
            raise ValueError(
                f"backend={backend!r} not in {sorted(_ALLOWED_BACKENDS)}"
            )

        host, port, index_name = _parse_uri(uri)
        client = BinaryClient.connect(host, port)
        try:
            if create:
                try:
                    client.vindex_create(
                        index_name, dim, kind=kind, backend=backend
                    )
                except ServerError as e:
                    # The server emits "already exists" (or a close
                    # variant) when the VINDEX is present. Anything
                    # else is a real failure and should surface.
                    if "exists" not in str(e).lower():
                        raise
        except BaseException:
            # Construction failed after the connection was opened.
            # Close it before propagating so the caller does not
            # have to.
            client.close()
            raise

        store = cls(client=client, index_name=index_name, dim=dim)
        store._owns_client = True
        return store

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying TCP connection if the store owns it.

        A store constructed by :meth:`from_uri` owns its client; one
        constructed directly with an external `BinaryClient` does not,
        and `close()` is a no-op in that case so the caller's lifetime
        management is respected.
        """
        if self._owns_client:
            self.client.close()
            self._owns_client = False

    def __enter__(self) -> "SkegVectorStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ── LlamaIndex BasePydanticVectorStore protocol ─────────────────

    @property
    def client_handle(self) -> BinaryClient:
        """Underlying `BinaryClient`. Provided for diagnostic access;
        treat as read-only and do not share across threads."""
        return self.client

    def add(self, nodes: Sequence["BaseNode"], **_: Any) -> list[str]:
        """Insert or replace nodes. Returns the list of node_ids stored.

        Per-node cost: one `VSET` + two `SET`. The three writes are not
        atomic; if the second or third fails mid-stream the vector is
        in VINDEX but the KV reverse mapping is absent, in which case
        :meth:`query` skips the orphan vec_id silently. For bulk loads
        prefer the offline build path (`skeg-cli build`).
        """
        from llama_index.core.schema import MetadataMode

        if not nodes:
            return []

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

            # Reverse map + payload. MetadataMode.NONE strips metadata
            # from the text body itself; the metadata dict is stored
            # separately in the same blob.
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            blob = _serialize_blob(text, dict(node.metadata or {}))
            self.client.set(
                _nid_key(vec_id), node.node_id.encode("utf-8")
            )
            self.client.set(_meta_key(node.node_id), blob)
            stored.append(node.node_id)
        return stored

    def delete(self, ref_doc_id: str, **_: Any) -> None:
        """Delete a node. The protocol calls the argument `ref_doc_id`;
        the value is the same as the `node_id` used at `add()` time."""
        vec_id = _node_id_to_vec_id(ref_doc_id)
        # Three best-effort drops. Missing entries return False which
        # we ignore: a delete of a never-added node is a no-op.
        self.client.vdel(self.index_name, vec_id)
        self.client.delete(_nid_key(vec_id))
        self.client.delete(_meta_key(ref_doc_id))

    def get_nodes(self, node_ids: Sequence[str]) -> list["TextNode | None"]:
        """Rehydrate `TextNode` objects from stored text + metadata.

        Order matches `node_ids`. A missing id yields `None` in that
        slot. One round-trip via `mget`.
        """
        from llama_index.core.schema import TextNode

        if not node_ids:
            return []
        keys = [_meta_key(nid) for nid in node_ids]
        blobs = self.client.mget(keys)
        out: list[TextNode | None] = []
        for nid, blob in zip(node_ids, blobs):
            if blob is None:
                out.append(None)
                continue
            text, meta = _deserialize_blob(blob)
            out.append(TextNode(id_=nid, text=text, metadata=meta))
        return out

    def query(
        self, query: "VectorStoreQuery", **_: Any
    ) -> "VectorStoreQueryResult":
        """Run a top-k similarity query and return ids, similarities,
        and rehydrated `TextNode` objects."""
        from llama_index.core.schema import TextNode
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
            self.index_name, list(query.query_embedding), k=k
        )
        if not hits:
            return VectorStoreQueryResult(ids=[], similarities=[], nodes=[])

        # Round-trip 1: u64 -> node_id reverse lookup, batched.
        nid_blobs = self.client.mget([_nid_key(h.id) for h in hits])

        node_ids: list[str] = []
        similarities: list[float] = []
        live_indices: list[int] = []
        for idx, (hit, raw) in enumerate(zip(hits, nid_blobs)):
            if raw is None:
                # vec_id in VINDEX but no KV mapping; the row is
                # half-deleted or never fully written. Skip.
                continue
            node_ids.append(raw.decode("utf-8"))
            similarities.append(hit.score)
            live_indices.append(idx)

        if not node_ids:
            return VectorStoreQueryResult(ids=[], similarities=[], nodes=[])

        # Round-trip 2: node_id -> blob lookup, also batched.
        blobs = self.client.mget([_meta_key(nid) for nid in node_ids])
        nodes: list[TextNode] = []
        for nid, blob in zip(node_ids, blobs):
            if blob is None:
                # Reverse mapping survived but payload is gone. Hand
                # back an empty TextNode rather than dropping the row;
                # the caller still sees the id + similarity.
                nodes.append(TextNode(id_=nid, text="", metadata={}))
                continue
            text, meta = _deserialize_blob(blob)
            nodes.append(TextNode(id_=nid, text=text, metadata=meta))

        return VectorStoreQueryResult(
            ids=node_ids, similarities=similarities, nodes=nodes
        )

    # ── Diagnostics ──────────────────────────────────────────────────

    def lookup_text(self, node_id: str) -> tuple[str, dict[str, Any]] | None:
        """Return `(text, metadata)` for a stored node, or `None` if
        the node is not present. Convenience for tests and ad-hoc
        inspection; production code should use :meth:`get_nodes`."""
        blob = self.client.get(_meta_key(node_id))
        if blob is None:
            return None
        return _deserialize_blob(blob)
