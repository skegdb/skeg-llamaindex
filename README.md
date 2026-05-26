# skeg-llamaindex

LlamaIndex `VectorStore` adapter for [skeg](https://github.com/skegdb/skeg).
Compatible with `llama-index-core >= 0.10`.

## Install

```sh
pip install skeg-llamaindex
```

Pulls in `skeg` (Python client) and `llama-index-core` automatically.
Server install (skeg engine) is separate:

```sh
brew tap skegdb/tap
brew install skeg
```

## Usage

```python
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.node_parser import SentenceSplitter
from skeg_llamaindex import SkegVectorStore

# 1. Start a skeg server:
#    skeg --data-dir ./data --addr 127.0.0.1:7379

# 2. Point the adapter at it. `dim` must match the embedding model.
store = SkegVectorStore.from_uri(
    "skeg://127.0.0.1:7379/notes",
    dim=1024,             # mxbai-embed-large-v1 dimension
    kind="int8",          # tier-1 quantisation
    backend="flat",       # in-RAM flat index for <50K vectors
)

# 3. Wire into LlamaIndex.
ctx = StorageContext.from_defaults(vector_store=store)
docs = [Document(text="hello world"), Document(text="goodbye world")]
index = VectorStoreIndex.from_documents(docs, storage_context=ctx)

# 4. Query as usual.
engine = index.as_query_engine()
print(engine.query("what does the first doc say?"))
```

## Index backend choice

| Use case | `backend` | Notes |
| --- | --- | --- |
| Personal AI, < 50K nodes | `flat` | Exhaustive scan; fast on M-series CPUs |
| RAG over a fixed corpus, > 50K nodes | `disk_vamana` (pre-build) | Use `skeg-tool build`, then serve read-only |
| Streaming insert with eventual large size | `disk_vamana` (RW) | Delta WAL handles streaming |

For the pre-build path, build offline once and start the server in
serve mode; this adapter then queries it read-only:

```sh
# Build the index offline (one shell):
skeg-cli build --input embeddings.npy --output ./data --name notes

# Serve it read-only (another shell):
skeg --mode serve --data-dir ./data --tier pq:128:256
```

## What this adapter handles

- `add(nodes)`: VSET each embedding + KV-store the text + metadata
- `query(VectorStoreQuery)`: VSEARCH top-k, returns node_ids + similarity
- `delete(ref_doc_id)`: VDEL + drop KV keys
- Stable mapping `node_id (str) → vec_id (u64)` via xxh3

## What this adapter does not do

- Metadata filter pushdown: LlamaIndex post-filters returned hits.
- Batched VSET: one VSET per node on the synchronous path. For large
  corpora build the index offline with `skeg-cli build` and serve it
  read-only.
- Hybrid sparse+dense search: skeg's surface is dense-only.
- Async API: this adapter is synchronous.

## Test-suite safety

The pytest suite spawns its own skeg-server via the conftest fixture
and tears it down at the end. The tests create VINDEX entries with
names like `notes-<test_name>` and drop them after each test. If you
ever override the fixture to point at an external server, those
VINDEX names may collide with yours. The fixture is the safe default.

## License

Apache-2.0.
