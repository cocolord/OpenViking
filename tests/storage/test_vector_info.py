# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Vector diagnostics must describe the loaded index and its actual dense scores."""

import json
from unittest.mock import Mock

import pytest

from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.vectordb.collection.collection import Collection
from openviking.storage.vectordb_adapters.base import CollectionAdapter
from openviking.storage.vectordb_adapters.local_adapter import (
    CuVSCollectionAdapter,
    LocalCollectionAdapter,
)
from openviking.storage.vectordb_adapters.opengauss.collection import OpenGaussCollection
from openviking.storage.vectordb_adapters.opengauss.sql import _distance_to_similarity
from openviking.storage.vectordb_adapters.opengauss_adapter import OpenGaussCollectionAdapter
from openviking.storage.viking_vector_index_backend import _SingleAccountBackend
from openviking_cli.utils.config.vectordb_config import VectorDBBackendConfig
from tests.vectordb.test_cuvs_collection import patch_cuvs_runtime


@pytest.mark.parametrize("mode", ["local", "cuvs"])
@pytest.mark.parametrize("sparse_weight", [0.0, 0.5])
@pytest.mark.parametrize(
    "metric,scale,score_range,scores",
    [
        ("cosine", "cosine_affine_0_1", [0.0, 1.0], [1.0, 0.5, 0.0]),
        ("ip", "inner_product", None, [2.0, 0.0, -2.0]),
        ("l2", "one_minus_squared_l2", [None, 1.0], [0.0, -1.0, -8.0]),
    ],
)
def test_loaded_vector_info_matches_dense_scores(
    tmp_path, monkeypatch, mode, sparse_weight, metric, scale, score_range, scores
):
    adapter: LocalCollectionAdapter
    if mode == "cuvs":
        patch_cuvs_runtime(monkeypatch)
        adapter = CuVSCollectionAdapter(
            "context", str(tmp_path), "default", {"algorithm": "brute_force"}
        )
    else:
        adapter = LocalCollectionAdapter("context", str(tmp_path), "default")
    try:
        assert adapter.get_vector_info() is None
        assert adapter.create_collection(
            "context",
            CollectionSchemas.context_collection("context", 4),
            distance=metric,
            sparse_weight=sparse_weight,
            index_name="default",
        )
        adapter.upsert(
            [
                {"id": "parallel", "vector": [2, 0, 0, 0]},
                {"id": "orthogonal", "vector": [0, 1, 0, 0]},
                {"id": "opposite", "vector": [-2, 0, 0, 0]},
            ]
        )
        expected_info = {
            "backend": mode,
            "collection_name": "context",
            "index_name": "default",
            "distance_metric": metric,
            "dense_score": {
                "scale": scale,
                "range": score_range,
                "higher_is_better": True,
            },
        }
        for _ in range(2):
            before = adapter.query(query_vector=[1, 0, 0, 0], limit=3)
            assert adapter.get_vector_info() == expected_info
            # Introspection must neither rescale scores nor modify search results.
            after = adapter.query(query_vector=[1, 0, 0, 0], limit=3)
            assert after == before
            assert [hit["id"] for hit in after] == ["parallel", "orthogonal", "opposite"]
            assert [hit["_score"] for hit in after] == pytest.approx(scores)
            adapter.close()  # Next iteration loads the persisted int8 index.

        metadata_files = list(tmp_path.rglob("index_meta.json"))
        assert len(metadata_files) == 1
        stored = json.loads(metadata_files[0].read_text())["VectorIndex"]
        assert stored["Distance"] == ("ip" if metric == "cosine" else metric)
        assert stored["NormalizeVector"] is (metric == "cosine")
    finally:
        adapter.close()


async def test_vector_info_uses_persisted_metric_after_config_change(tmp_path):
    config = VectorDBBackendConfig(
        backend="local", path=str(tmp_path), dimension=4, distance_metric="cosine"
    )
    backend = _SingleAccountBackend(config, bound_account_id="account-a")
    try:
        assert await backend.create_collection(
            "context", CollectionSchemas.context_collection("context", 4)
        )
    finally:
        await backend.close()

    backend = _SingleAccountBackend(
        config.model_copy(update={"distance_metric": "ip"}), bound_account_id="account-a"
    )
    try:
        info = await backend.get_vector_info()
        assert info["distance_metric"] == "cosine"
        assert info["dense_score"]["scale"] == "cosine_affine_0_1"
    finally:
        await backend.close()


class _RemoteAdapter(CollectionAdapter):
    mode = "http"

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError

    def _load_existing_collection_if_needed(self):
        pass

    def _create_backend_collection(self, meta):
        raise NotImplementedError


@pytest.mark.parametrize(
    "metadata,metric",
    [
        ({"IndexName": "default"}, None),
        ({"VectorIndex": {"Distance": "cosine"}}, "cosine"),
        ({"VectorIndex": {"Distance": "ip", "NormalizeVector": True}}, "cosine"),
        ({"VectorIndex": {"Distance": "ip", "NormalizeVector": False}}, "ip"),
        ({"VectorIndex": {"Distance": "L2"}}, "l2"),
        ({"VectorIndex": {"Distance": "vendor_metric"}}, "vendor_metric"),
    ],
)
def test_remote_metadata_does_not_imply_local_score_scale(metadata, metric):
    adapter = _RemoteAdapter("context", "default")
    adapter._collection = Mock(spec=Collection)
    adapter._collection.get_index_meta_data.return_value = metadata
    info = adapter.get_vector_info()
    assert info is not None
    assert info["distance_metric"] == metric
    assert info["dense_score"] == {
        "scale": "backend_defined",
        "range": None,
        "higher_is_better": None,
    }
    adapter._collection.get_index_meta_data.assert_called_once_with("default")


def test_unsupported_metadata_is_unknown_but_backend_failures_propagate():
    adapter = _RemoteAdapter("context", "default")
    adapter._collection = Mock(spec=Collection)
    adapter._collection.get_index_meta_data.side_effect = NotImplementedError
    info = adapter.get_vector_info()
    assert info is not None
    assert info["distance_metric"] is None
    adapter._collection.get_index_meta_data.side_effect = RuntimeError("backend unavailable")
    with pytest.raises(RuntimeError, match="backend unavailable"):
        adapter.get_vector_info()


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize(
    "index_meta,metric,scale,score_range,distance,score",
    [
        ({"_distance": "ip", "Distance": "cosine"}, "ip", "inner_product", None, -2, 2),
        ({"Distance": "l2"}, "l2", "inverse_one_plus_distance", [0.0, 1.0], 3, 0.25),
        ({"_distance": "l1"}, "l1", "inverse_one_plus_distance", [0.0, 1.0], 4, 0.2),
        ({}, "cosine", "cosine_similarity", [-1.0, 1.0], 1.5, -0.5),
    ],
)
def test_opengauss_info_matches_query_metric_resolution(
    pending, index_meta, metric, scale, score_range, distance, score
):
    # Exercise the real collection's metadata and query metric resolver without a database.
    collection = object.__new__(OpenGaussCollection)
    collection._distance = "cosine"
    collection._meta = {"_distance": "cosine"}
    collection._indexes = {} if pending else {"default": index_meta}
    collection._pending_indexes = {"default": index_meta} if pending else {}
    adapter = object.__new__(OpenGaussCollectionAdapter)
    CollectionAdapter.__init__(adapter, "context", "default")
    adapter._collection = Collection(collection)
    info = adapter.get_vector_info()
    assert info is not None
    assert info["distance_metric"] == collection._resolve_distance_and_op("default")[0] == metric
    assert info["dense_score"] == {"scale": scale, "range": score_range, "higher_is_better": True}
    assert _distance_to_similarity(info["distance_metric"], distance) == pytest.approx(score)
