# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Backend request translation and HTTP transport for native decay."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openviking.storage.vectordb.collection.result import SearchItemResult, SearchResult
from openviking.storage.vectordb_adapters.local_adapter import LocalCollectionAdapter
from openviking.storage.vectordb_adapters.opengauss.collection import OpenGaussCollection


@pytest.mark.parametrize("mode", ["local", "cuvs", "http", "vikingdb", "volcengine"])
def test_adapter_owns_decay_parameters_and_keeps_requested_window(mode):
    adapter = LocalCollectionAdapter("context", "", "default")
    adapter.mode = mode
    coll = Mock()
    coll.search_by_vector.return_value = SearchResult(
        data=[SearchItemResult(id="one", score=0.8, fields={"updated_at": "2026-01-01T00:00:00Z"})]
    )
    adapter.get_collection = lambda: coll
    adapter.query(
        query_vector=[1],
        limit=10,
        offset=2,
        advance={
            "time_decay": {
                "protection": "0",
                "origin": "2026-01-08T00:00:00Z",
            }
        },
    )
    kwargs = coll.search_by_vector.call_args.kwargs
    assert kwargs["limit"] == 10 and kwargs["offset"] == 2
    if mode in {"vikingdb", "volcengine"}:
        assert set(kwargs["advance"]) == {"post_process_ops"}
        assert kwargs["advance"]["post_process_ops"][0]["fusion_by"] == "multiply"
    else:
        rule = kwargs["advance"]["time_decay"]
        assert rule["offset_ms"] == 0 and rule["scale_ms"] == 7 * 24 * 3600 * 1000
    assert kwargs["return_detail_info"] is True


@pytest.mark.asyncio
async def test_http_transport_delivers_native_rule_and_score_details(monkeypatch):
    from openviking.storage.vectordb.collection.http_collection import HttpCollection
    from openviking.storage.vectordb.service import api_fastapi
    from openviking.storage.vectordb.service.app_models import SearchByVectorRequest

    native_rule = {
        "time_decay": {
            "field": "updated_at",
            "origin_ms": 1767830400000,
            "offset_ms": 0,
            "scale_ms": 604800000,
            "decay": 0.5,
        }
    }
    coll = Mock()
    coll.search_by_vector.return_value = SearchResult(
        data=[SearchItemResult(id="one", score=0.4, origin_score=0.8, addition_score=0.5)]
    )
    monkeypatch.setattr(api_fastapi, "get_collection_or_raise", lambda *args: coll)
    request = SearchByVectorRequest(
        collection_name="context",
        index_name="default",
        dense_vector=[1],
        advance=native_rule,
        return_detail_info=True,
    )
    await api_fastapi.search_by_vector(request, SimpleNamespace(state=SimpleNamespace()))
    assert coll.search_by_vector.call_args.kwargs["advance"] == native_rule
    assert coll.search_by_vector.call_args.kwargs["return_detail_info"] is True
    client = HttpCollection.__new__(HttpCollection)
    client.url_prefix, client.project_name, client.collection_name = (
        "http://localhost/",
        "default",
        "context",
    )
    post = Mock(
        return_value=SimpleNamespace(
            status_code=200,
            text='{"data":{"data":[{"id":"one","score":0.4,"origin_score":0.8,"addition_score":0.5}]}}',
        )
    )
    monkeypatch.setattr(
        "openviking.storage.vectordb.collection.http_collection.requests.post", post
    )
    result = client.search_by_vector(
        "default", dense_vector=[1], limit=10, advance=native_rule, return_detail_info=True
    )
    assert post.call_args.kwargs["json"]["advance"] == native_rule
    assert result.data[0].origin_score == 0.8 and result.data[0].addition_score == 0.5


def test_opengauss_ranks_in_sql_before_joining_payloads():
    coll = OpenGaussCollection.__new__(OpenGaussCollection)
    coll._name, coll._dim, coll._distance, coll._distributed = "context", 2, "cosine", False
    coll._field_names = {"id", "vector", "updated_at", "abstract"}
    coll._date_time_fields, coll._array_fields = {"updated_at"}, set()
    coll._indexes, coll._pending_indexes = {"default": {"_distance": "cosine"}}, {}
    coll._lock, coll._conn = threading.RLock(), Mock()
    coll._materialize_pending_index = Mock()
    coll.has_index = Mock(return_value=True)
    coll._apply_search_params_on_cursor = Mock()
    cursor = coll._conn.cursor.return_value
    cursor.description = [
        ("id",),
        ("abstract",),
        ("_origin_score",),
        ("_time_score",),
        ("_final_score",),
    ]
    cursor.fetchall.return_value = [("one", "retained", 0.8, 0.5, 0.4)]
    result = coll.search_by_vector(
        "default",
        dense_vector=[1, 0],
        limit=10,
        offset=2,
        output_fields=["abstract"],
        advance={
            "time_decay": {
                "field": "updated_at",
                "origin_ms": 1767830400000,
                "offset_ms": 0,
                "scale_ms": 604800000,
                "decay": 0.5,
            }
        },
    )
    sql, params = cursor.execute.call_args.args
    assert params[1] == 36
    assert '"abstract"' not in sql.split("scored AS")[0]
    assert "(_origin_score * _time_score) AS _final_score" in sql
    assert "JOIN payloads" in sql
    assert result.data[0].score == 0.4
    assert result.data[0].fields == {"abstract": "retained"}
