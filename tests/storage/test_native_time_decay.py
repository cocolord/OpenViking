# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Native time decay ranks bounded candidates before fetching result payloads."""

from datetime import datetime, timezone

import pytest

from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.expr import Eq
from openviking.utils.time_decay import build_time_decay_fusion_spec
from openviking_cli.utils.config.vectordb_config import VectorDBBackendConfig
from tests.storage.test_viking_vector_scope_filter import _ctx


@pytest.mark.asyncio
async def test_native_decay_bounds_candidates_and_fetches_only_final_topk(
    vector_backend_factory, tmp_path, monkeypatch
):
    backend = vector_backend_factory(
        config=VectorDBBackendConfig(
            backend="local", name="context", dimension=4, path=str(tmp_path)
        )
    )
    ctx = _ctx()
    try:
        await backend.create_collection(
            "context", CollectionSchemas.context_collection("context", 4)
        )
        records = []
        for i in range(50):
            records.append(
                {
                    "id": str(i),
                    "uri": f"viking://user/alice/memories/events/{i}.md",
                    "account_id": "acct",
                    "context_type": "memory",
                    "level": 2,
                    "abstract": "payload " + str(i),
                    "search_tags": ["memory_type=events"],
                    "updated_at": "2026-01-08T00:00:00Z" if i >= 10 else "2025-01-01T00:00:00Z",
                    "vector": [1 - i * 0.01, (1 - (1 - i * 0.01) ** 2) ** 0.5, 0, 0],
                }
            )
        await backend._upsert_many_raw(records, ctx=ctx)
        account = await backend._get_backend_for_context(ctx)
        collection = account._adapter.get_collection()
        # Public wrapper delegates to the local collection.
        local = collection._Collection__collection
        fetched = []
        original = local.store_mgr.fetch_cands_fields

        def fetch(labels):
            fetched.append(list(labels))
            return original(labels)

        monkeypatch.setattr(local.store_mgr, "fetch_cands_fields", fetch)
        results = await backend.search_in_tenant(
            ctx=ctx,
            query_vector=[1, 0, 0, 0],
            context_type="memory",
            level=[2],
            limit=10,
            events_time_decay_protection="0",
            request_now=datetime(2026, 1, 8, tzinfo=timezone.utc),
        )
        assert [r["id"] for r in results] == [str(i) for i in range(10, 20)]
        assert sum(map(len, fetched)) == 10
        assert all(r["_time_score"] == pytest.approx(1) for r in results)
        # Top-1's internal budget is 3. It intentionally cannot promote rank 11.
        one = await backend.search_in_tenant(
            ctx=ctx,
            query_vector=[1, 0, 0, 0],
            context_type="memory",
            level=[2],
            limit=1,
            events_time_decay_protection="0",
            request_now=datetime(2026, 1, 8, tzinfo=timezone.utc),
        )
        assert one[0]["id"] == "0"
        assert one[0]["_score"] < 0.001
        # A retriever requests its own rerank window. Native attaches time scores
        # while preserving semantic order and does not expand/truncate that window.
        deferred = await backend.search_in_tenant(
            ctx=ctx,
            query_vector=[1, 0, 0, 0],
            context_type="memory",
            level=[2],
            limit=3,
            for_rerank=True,
            events_time_decay_protection="0",
            request_now=datetime(2026, 1, 8, tzinfo=timezone.utc),
        )
        assert [r["id"] for r in deferred] == ["0", "1", "2"]
        assert deferred[0]["_score"] == pytest.approx(1)
        assert deferred[0]["_time_score"] < 0.001
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("protection", ["0", "2d", "7d"])
async def test_native_decay_matches_reference_with_offset_and_missing_time(
    vector_backend_factory, tmp_path, protection
):
    backend = vector_backend_factory(
        config=VectorDBBackendConfig(
            backend="local", name="context", dimension=4, path=str(tmp_path)
        )
    )
    ctx = _ctx()
    origin = datetime(2026, 1, 8, tzinfo=timezone.utc)
    times = ["2026-01-01T00:00:00Z", "2026-01-08T00:00:00Z", "2026-01-15T00:00:00Z", None]
    try:
        await backend.create_collection(
            "context", CollectionSchemas.context_collection("context", 4)
        )
        for i, timestamp in enumerate(times):
            data = {
                "id": str(i),
                "uri": f"viking://user/alice/memories/events/{i}",
                "account_id": "acct",
                "context_type": "memory",
                "level": 2,
                "vector": [1, 0, 0, 0],
                "search_tags": ["memory_type=events"],
            }
            if timestamp:
                data["updated_at"] = timestamp
            await backend._upsert_many_raw([data], ctx=ctx)
        result = await backend.search(
            ctx=ctx,
            query_vector=[1, 0, 0, 0],
            filter=Eq("level", 2),
            limit=2,
            offset=1,
            advance={"time_decay": {"protection": protection, "origin": origin.isoformat()}},
        )
        spec = build_time_decay_fusion_spec(protection=protection, origin=origin)
        expected_scores = sorted([spec.fuse_optional(1, t)[0] for t in times], reverse=True)[1:3]
        assert [r["_score"] for r in result] == pytest.approx(expected_scores)
        for item in result:
            expected = spec.fuse_optional(1, times[int(item["id"])])
            assert item["_score"] == pytest.approx(expected[0])
            assert (
                item.get("_time_score") == pytest.approx(expected[1])
                if expected[1] is not None
                else "_time_score" not in item
            )
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_native_decay_survives_restart_and_timestamp_update(vector_backend_factory, tmp_path):
    config = VectorDBBackendConfig(backend="local", name="context", dimension=4, path=str(tmp_path))
    ctx = _ctx()
    backend = vector_backend_factory(config=config)
    await backend.create_collection("context", CollectionSchemas.context_collection("context", 4))
    for i, date in enumerate(["2026-01-01T00:00:00Z", "2026-01-08T00:00:00Z"]):
        await backend._upsert_many_raw(
            [
                {
                    "id": str(i),
                    "uri": f"viking://user/alice/memories/events/{i}",
                    "account_id": "acct",
                    "context_type": "memory",
                    "level": 2,
                    "vector": [1, 0, 0, 0],
                    "updated_at": date,
                    "search_tags": ["memory_type=events"],
                }
            ],
            ctx=ctx,
        )
    await backend.close()
    reopened = vector_backend_factory(config=config)
    kwargs = {
        "ctx": ctx,
        "query_vector": [1, 0, 0, 0],
        "context_type": "memory",
        "level": [2],
        "limit": 1,
        "events_time_decay_protection": "0",
        "request_now": datetime(2026, 1, 8, tzinfo=timezone.utc),
    }
    try:
        result = await reopened.search_in_tenant(**kwargs)
        assert result[0]["id"] == "1"
        await reopened._upsert_many_raw(
            [
                {
                    "id": "1",
                    "uri": "viking://user/alice/memories/events/1",
                    "account_id": "acct",
                    "context_type": "memory",
                    "level": 2,
                    "vector": [1, 0, 0, 0],
                    "updated_at": "2025-01-01T00:00:00Z",
                    "search_tags": ["memory_type=events"],
                }
            ],
            ctx=ctx,
        )
        result = await reopened.search_in_tenant(**kwargs)
        assert result[0]["id"] == "0"
        assert result[0]["_time_score"] == pytest.approx(0.5)
    finally:
        await reopened.close()
