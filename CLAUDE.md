# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv run pytest                          # run all tests
uv run pytest tests/test_store.py::test_add_and_search  # run single test
uv run python -c "import baked; print(baked.__version__)"  # smoke check
```

No linter configured yet.

## Architecture

Single-class library. All logic lives in `baked/store.py`. `baked/__init__.py` re-exports `VectorStore` only.

### Storage: two files per store

| File | Format | Purpose |
|------|--------|---------|
| `<path>.vecs` | binary | Header: `II` (dim, count as uint32). Then N × dim × 4-byte floats (native endian). |
| `<path>.json` | JSON array | One metadata dict per vector. Index in array = index in `.vecs`. |

Both files are written atomically together on `save()`. If only one exists, the store treats itself as empty (no partial load).

### Key invariants

- **All stored vectors are L2-normalized** at `add()` time. Search scores are therefore dot products, not raw cosine similarity — do not compare scores across stores with different normalization states.
- **Deletes are soft** (`_deleted: set[int]`) until `save()`, which compacts both files and resets in-memory state. IDs shift after compaction — do not hold IDs across a `save()` call that follows a `remove()`.
- **Dim is inferred from the first `add()` call** and locked for the lifetime of the store. Mismatches raise `ValueError`.

### Search

`search()` is brute-force full scan every call — no index. Filter (`dict`) is checked before scoring: all key-value pairs must match the metadata dict exactly. `radius` mode returns all vectors with score ≥ threshold; `top_k` mode returns the top N sorted results.
