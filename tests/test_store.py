import math
import os
import tempfile
import pytest
from baked import VectorStore


def make_vec(angle_deg: float, dim: int = 4) -> list[float]:
    """Unit vector with first two components set by angle, rest zero."""
    rad = math.radians(angle_deg)
    v = [math.cos(rad), math.sin(rad)] + [0.0] * (dim - 2)
    return v


def test_add_and_search(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    store.add(make_vec(0), {"label": "right"})
    store.add(make_vec(90), {"label": "up"})
    store.add(make_vec(45), {"label": "diagonal"})

    results = store.search(make_vec(0), top_k=2)
    assert results[0][0]["label"] == "right"
    assert results[1][0]["label"] == "diagonal"


def test_save_and_reload(tmp_path):
    path = str(tmp_path / "db")
    store = VectorStore(path)
    store.add(make_vec(0), {"label": "right"})
    store.add(make_vec(90), {"label": "up"})
    store.save()

    store2 = VectorStore(path)
    assert len(store2) == 2
    results = store2.search(make_vec(90), top_k=1)
    assert results[0][0]["label"] == "up"


def test_remove_and_save(tmp_path):
    path = str(tmp_path / "db")
    store = VectorStore(path)
    store.add(make_vec(0), {"label": "right"})
    store.add(make_vec(90), {"label": "up"})
    store.remove(0)
    store.save()

    store2 = VectorStore(path)
    assert len(store2) == 1
    results = store2.search(make_vec(0), top_k=5)
    assert all(r[0]["label"] != "right" for r in results)


def test_filter(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    store.add(make_vec(0), {"source": "a", "label": "right"})
    store.add(make_vec(10), {"source": "b", "label": "near-right"})
    store.add(make_vec(90), {"source": "a", "label": "up"})

    results = store.search(make_vec(0), top_k=5, filter={"source": "a"})
    labels = [r[0]["label"] for r in results]
    assert "near-right" not in labels
    assert "right" in labels


def test_range_search(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    store.add(make_vec(0), {"label": "right"})
    store.add(make_vec(10), {"label": "near"})
    store.add(make_vec(89), {"label": "almost-up"})

    # cos(89°) ≈ 0.017 — only tight matches at radius=0.9
    results = store.search(make_vec(0), radius=0.9)
    labels = [r[0]["label"] for r in results]
    assert "right" in labels
    assert "almost-up" not in labels


def test_batch_search(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    store.add(make_vec(0), {"label": "right"})
    store.add(make_vec(90), {"label": "up"})

    results = store.search_many([make_vec(0), make_vec(90)], top_k=1)
    assert results[0][0][0]["label"] == "right"
    assert results[1][0][0]["label"] == "up"


def test_dim_mismatch(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    store.add([1.0, 0.0], {})
    with pytest.raises(ValueError, match="Dimension mismatch"):
        store.add([1.0, 0.0, 0.0], {})


def test_zero_vector_raises(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    with pytest.raises(ValueError, match="zero vector"):
        store.add([0.0, 0.0, 0.0], {})


def test_empty_search(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    results = store.search(make_vec(0), top_k=5)
    assert results == []


def test_save_empty_store_removes_files(tmp_path):
    path = str(tmp_path / "db")
    store = VectorStore(path)
    store.add(make_vec(0), {})
    store.save()
    store.remove(0)
    store.save()
    assert not os.path.exists(path + ".vecs")
    assert not os.path.exists(path + ".json")
