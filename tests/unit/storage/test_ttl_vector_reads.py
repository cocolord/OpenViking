# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""TTL reads work against the pre-TTL vector schema, including candidate refill."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.core.ttl import TTL_FIELD_NAMES
from openviking.pyagfs.exceptions import AGFSNetworkError
from openviking.server.identity import RequestContext, Role
from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.expr import Eq
from openviking.storage.ovpack.index import EXPORT_VECTOR_FIELDS
from openviking.storage.ttl_registry import TTLRegistry
from openviking.storage.vectordb.index.cuvs_index import matches_filter
from openviking.storage.vectordb_adapters.local_adapter import LocalCollectionAdapter
from openviking.storage.viking_fs import VikingFS
from openviking.storage.viking_vector_index_backend import (
    FETCH_BY_URI_OUTPUT_FIELDS,
    RETRIEVAL_OUTPUT_FIELDS,
    VikingVectorIndexBackend,
    _SingleAccountBackend,
)
from openviking_cli.session.user_id import UserIdentifier
from tests.unit.storage.ttl_test_storage import MemoryAGFS

ROOT = "viking://user/alice/memories/events"
PAST = "2000-01-01T00:00:00.000Z"
FUTURE = "2999-01-01T00:00:00.000Z"


def test_vector_schema_and_projections_need_no_ttl_columns():
    schema = CollectionSchemas.context_collection("context", 2)
    names = {item["FieldName"] for item in schema["Fields"]}
    for fields in (
        names,
        schema["ScalarIndex"],
        RETRIEVAL_OUTPUT_FIELDS,
        FETCH_BY_URI_OUTPUT_FIELDS,
        EXPORT_VECTOR_FIELDS,
    ):
        assert TTL_FIELD_NAMES.isdisjoint(fields)


def test_vector_writes_strip_lifecycle_fields_even_without_schema_metadata():
    backend = object.__new__(_SingleAccountBackend)
    backend._filter_known_fields = lambda data: data
    backend._adapter = SimpleNamespace(USE_CONTENT_FIELD=False)
    record = {
        "uri": ROOT + "/2026/09/02/event.md",
        "level": 2,
        "expires_at": FUTURE,
        "ttl_generation": "incarnation-1",
        "received_at": PAST,
        "ttl_days": 2,
    }
    assert backend._prepare_upsert_payload(record) == {"uri": record["uri"], "level": 2}
    assert record["ttl_generation"] == "incarnation-1"


@pytest.fixture
def setup(monkeypatch):
    ctx = RequestContext(user=UserIdentifier("acct", "alice"), role=Role.ROOT)
    fs = VikingFS(agfs=SimpleNamespace())
    fs._async_agfs = MemoryAGFS()
    files = fs._async_agfs.files
    fs._async_agfs.read = AsyncMock(wraps=fs._async_agfs.read)
    fs.ttl_registry.account_may_have_records = AsyncMock(return_value=True)
    monkeypatch.setattr("openviking.storage.viking_fs.get_viking_fs", lambda: fs)
    monkeypatch.setattr("openviking.storage.viking_vector_index_backend.ttl_enabled", lambda: False)

    def source(uri, expiry=None):
        fields = {"expires_at": expiry} if expiry else {}
        files[fs._uri_to_path(uri, ctx=ctx)] = b"body"
        files[fs._uri_to_path(uri.rsplit("/", 1)[0] + "/.meta.json", ctx=ctx)] = json.dumps(
            fields
        ).encode()

    rows, calls = [], []
    compiler = object.__new__(LocalCollectionAdapter)

    async def query(**kwargs):
        compiled = compiler._compile_filter(kwargs["filter"])
        assert "expires_at" not in json.dumps(compiled)
        assert TTL_FIELD_NAMES.isdisjoint(kwargs.get("output_fields") or [])
        calls.append(kwargs)
        selected = [
            row
            for row in rows
            if matches_filter(
                {**row, "uri": compiler._encode_uri_field_value(row["uri"])},
                compiled,
                {"uri": "path", "level": "int64", "account_id": "string"},
            )
        ]
        if order_by := kwargs.get("order_by"):
            selected.sort(key=lambda row: row[order_by], reverse=kwargs.get("order_desc", False))
        start = kwargs.get("offset", 0)
        fields = kwargs.get("output_fields")
        page_limit = min(kwargs["limit"], getattr(single, "page_limit", kwargs["limit"]))
        return [
            {
                key: value
                for key, value in row.items()
                if fields is None or key in fields or key == "_score"
            }
            for row in selected[start : start + page_limit]
        ]

    single = SimpleNamespace(query=query, search_by_random=query, search_by_keywords=query)
    backend = object.__new__(VikingVectorIndexBackend)
    backend.acl_manager = None
    backend._get_backend_for_context = AsyncMock(return_value=single)
    return SimpleNamespace(
        ctx=ctx,
        fs=fs,
        files=files,
        source=source,
        rows=rows,
        calls=calls,
        backend=backend,
        single=single,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method",
    [
        "search_in_tenant",
        "filter_in_tenant",
        "search_by_keywords",
        "search_by_random",
        "query",
    ],
)
async def test_expired_candidates_are_replaced_before_limit(setup, method):
    s = setup
    for i in range(7):
        uri = f"{ROOT}/2026/09/{i + 1:02}/{i}.md"
        s.source(uri, PAST if i < 3 else FUTURE)
        s.rows.append({"uri": uri, "level": 2, "account_id": "acct", "_score": 1 - i / 10})
    kwargs = {"ctx": s.ctx, "limit": 2}
    if method == "search_in_tenant":
        kwargs["query_vector"] = [0.1, 0.2]
    if method == "filter_in_tenant":
        kwargs["target_directories"] = [ROOT]
    if method == "search_by_keywords":
        kwargs.update(query="meeting", mode="bm25", fields=["content"])
    if method == "query":
        # A remote backend may cap refill batches above the original page size.
        s.single.page_limit = 2
        kwargs.update(
            query_vector=[0.1, 0.2],
            include_expired=False,
            advance={"time_decay": {"protection": "0", "origin": "2026-09-30T00:00:00Z"}},
        )
    result = await getattr(s.backend, method)(**kwargs)
    assert [row["uri"] for row in result] == [f"{ROOT}/2026/09/04/3.md", f"{ROOT}/2026/09/05/4.md"]
    assert [call["limit"] for call in s.calls] == ([2, 4, 2] if method == "query" else [2, 4])
    assert [row["_score"] for row in result] == pytest.approx([0.7, 0.6])
    for key in ("advance", "mode", "fields"):
        if key in kwargs:
            assert all(call[key] == kwargs[key] for call in s.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("descending", [False, True])
async def test_offset_counts_live_rows_and_preserves_legacy_records(setup, descending):
    s = setup
    for i in range(9):
        uri = f"{ROOT}/2026/09/{i + 1:02}/{i}.md"
        s.source(uri, PAST if i < 3 or i > 5 else None)
        s.rows.append({"uri": uri, "level": 2, "updated_at": f"2026-09-{i + 1:02}T00:00:00Z"})
    result = await s.backend.filter(
        Eq("level", 2),
        limit=2,
        offset=1,
        output_fields=["updated_at"],
        order_by="updated_at",
        order_desc=descending,
        ctx=s.ctx,
        include_expired=False,
    )
    assert result == [
        {"updated_at": f"2026-09-{day:02}T00:00:00Z", "expires_at": None}
        for day in ([5, 4] if descending else [5, 6])
    ]
    assert [call["limit"] for call in s.calls] == [3, 6]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "root",
    [ROOT, "viking://resources/doc", ROOT + "/2026/09/01", "viking://user/alice/sessions/s1"],
)
async def test_summaries_follow_owner_expiry_while_containers_stay_visible(setup, root):
    s = setup
    from openviking.core.ttl import ttl_object_for_uri

    target = ttl_object_for_uri(root)
    if target:
        s.files[s.fs._uri_to_path(root, ctx=s.ctx)] = b"directory"
        s.files[s.fs._uri_to_path(root + "/.meta.json", ctx=s.ctx)] = json.dumps(
            {"expires_at": PAST}
        ).encode()
    for level in (0, 1):
        s.rows.append({"uri": root, "level": level, "abstract": "summary"})
    actual = await s.backend.query(ctx=s.ctx, include_expired=False)
    assert actual == ([] if target else [{**row, "expires_at": None} for row in s.rows])
    assert s.fs.ttl_registry.account_may_have_records.await_count == int(target is not None)


@pytest.mark.asyncio
async def test_live_owner_does_not_stat_each_vector_source(setup):
    s = setup
    owner = ROOT + "/2026/09/01"
    s.source(owner + "/body.md", FUTURE)
    # Leaf/summary existence is not part of directory TTL visibility.
    s.rows.extend(
        [{"uri": owner, "level": 0}, {"uri": owner, "level": 1}]
        + [{"uri": owner + f"/{i}.md", "level": 2} for i in range(20)]
    )
    result = await s.backend.query(ctx=s.ctx, limit=22, include_expired=False)
    assert len(result) == 22
    assert [call["limit"] for call in s.calls] == [22]
    assert s.fs._async_agfs.stat_calls == []
    assert s.fs._async_agfs.read.await_count == 1


@pytest.mark.asyncio
async def test_legacy_owner_is_checked_once_and_deleted_owner_is_hidden(setup):
    s = setup
    owner = ROOT + "/2026/09/01"
    body = owner + "/body.md"
    s.files[s.fs._uri_to_path(body, ctx=s.ctx)] = b"legacy"
    s.rows.extend([{"uri": owner + f"/{i}.md", "level": 2} for i in range(20)])
    assert len(await s.backend.query(ctx=s.ctx, limit=20, include_expired=False)) == 20
    assert s.fs._async_agfs.stat_calls.count(s.fs._uri_to_path(owner, ctx=s.ctx)) == 1
    del s.files[s.fs._uri_to_path(body, ctx=s.ctx)]
    assert await s.backend.query(ctx=s.ctx, limit=20, include_expired=False) == []


@pytest.mark.asyncio
async def test_refill_excludes_whole_expired_owner_across_levels_and_nested_files(setup):
    s = setup
    expired = ROOT + "/2026/09/01"
    live = ROOT + "/2026/09/10"  # A textual prefix must not exclude this sibling.
    s.source(expired + "/body.md", PAST)
    s.source(live + "/body.md", FUTURE)
    s.rows.extend(
        [{"uri": expired, "level": level} for level in (0, 1)]
        + [{"uri": expired + f"/nested/{i}.md", "level": 2} for i in range(20)]
        + [{"uri": live + f"/{i}.md", "level": 2} for i in range(2)]
    )
    result = await s.backend.query(ctx=s.ctx, limit=2, include_expired=False)
    assert [row["uri"] for row in result] == [live + "/0.md", live + "/1.md"]
    assert len(s.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [OSError("storage unavailable"), AGFSNetworkError("endpoint not found")]
)
@pytest.mark.parametrize("entrypoint", ["vector_query", "file_visibility"])
async def test_source_read_error_cannot_expose_content(setup, error, entrypoint):
    s = setup
    uri = ROOT + "/2026/09/02/event.md"
    s.source(uri, PAST)
    s.rows.append({"uri": uri, "level": 2})
    s.fs.ttl_registry = TTLRegistry(s.fs._async_agfs)
    s.fs._async_agfs.stat = AsyncMock(side_effect=error)
    s.fs._async_agfs.read = AsyncMock(side_effect=error)
    with pytest.raises(type(error), match=str(error)):
        if entrypoint == "vector_query":
            await s.backend.query(ctx=s.ctx, include_expired=False)
        else:
            await s.fs._ttl_uri_visible(uri, s.ctx)


@pytest.mark.asyncio
async def test_backend_ignoring_exclusion_fails_without_looping_forever(setup):
    s = setup
    row = {"uri": ROOT + "/2026/09/03/expired.md", "level": 2}
    s.source(row["uri"], PAST)
    s.single.query = AsyncMock(return_value=[row])
    with pytest.raises(RuntimeError, match="did not exclude"):
        await s.backend.query(limit=1, ctx=s.ctx, include_expired=False)
    assert s.single.query.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_during_query", [False, True])
async def test_unmanaged_reads_notice_ttl_enablement_even_with_default_off(
    setup, enable_during_query
):
    s = setup
    s.fs.ttl_registry = TTLRegistry(s.fs._async_agfs)
    s.rows.extend({"uri": f"{ROOT}/2026/09/{i:02}/body.md", "_score": i} for i in (1, 2, 3))
    advance = {"time_decay": {"protection": "0", "origin": "2026-09-30T00:00:00Z"}}
    assert await s.backend.query(
        ctx=s.ctx,
        limit=1,
        offset=1,
        output_fields=["_score"],
        advance=advance,
        include_expired=False,
    ) == [{"_score": 2, "expires_at": None}]
    assert s.calls[0]["advance"] == advance
    assert s.fs._async_agfs.stat_calls == [TTLRegistry.marker_path(s.ctx.account_id)] * 2
    assert (s.calls[0]["limit"], s.calls[0]["offset"]) == (1, 1)
    s.fs._async_agfs.read.assert_not_awaited()

    async def enable():
        # A different worker publishes the account marker before owner deadlines.
        await TTLRegistry(s.fs._async_agfs).mark_account(s.ctx.account_id)
        for row in s.rows:
            s.source(row["uri"], PAST)

    if enable_during_query:
        original = s.single.query

        async def query(**kwargs):
            records = await original(**kwargs)
            await enable()
            return records

        s.single.query = query
    else:
        await enable()
    assert await s.backend.query(ctx=s.ctx, include_expired=False) == []
    s.fs._async_agfs.read.assert_awaited()
    assert await s.backend.query(ctx=s.ctx) == s.rows


@pytest.mark.asyncio
async def test_count_uses_backend_total_until_physical_cleanup(setup):
    s = setup
    s.single.count = AsyncMock(return_value=2)
    assert await s.backend.count(ctx=s.ctx) == 2
    s.fs._async_agfs.read.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_offset_restarts_when_ttl_is_enabled_during_query(setup):
    s = setup
    s.fs.ttl_registry.account_may_have_records = AsyncMock(side_effect=[False, True, True])
    for i, expiry in ((1, PAST), (2, FUTURE), (3, FUTURE)):
        uri = f"{ROOT}/2026/09/{i:02}/body.md"
        s.rows.append({"uri": uri, "_score": i})
        s.source(uri, expiry)
    result = await s.backend.query(
        ctx=s.ctx, limit=1, offset=1, output_fields=["_score"], include_expired=False
    )
    assert result == [{"_score": 3, "expires_at": FUTURE}]
    assert [(call["limit"], call["offset"]) for call in s.calls] == [(1, 1), (2, 0), (2, 0)]


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", ["candidates", "owners"])
async def test_ttl_refill_budget_errors_instead_of_returning_partial_results(
    setup, monkeypatch, budget
):
    from openviking.storage import viking_vector_index_backend as module
    from openviking_cli.exceptions import ResourceExhaustedError

    s = setup
    monkeypatch.setattr(module, "_TTL_MAX_CANDIDATES", 6 if budget == "candidates" else 100)
    monkeypatch.setattr(module, "_TTL_MAX_EXCLUDED_OWNERS", 2 if budget == "owners" else 100)
    for i in range(1, 8):
        uri = f"{ROOT}/2026/09/{i:02}/body.md"
        s.rows.append({"uri": uri})
        s.source(uri, PAST if i < 7 else FUTURE)
    with pytest.raises(ResourceExhaustedError, match="TTL query budget exceeded"):
        await s.backend.query(ctx=s.ctx, limit=1, include_expired=False)
    assert [call["limit"] for call in s.calls] == [1, 2]


@pytest.mark.asyncio
async def test_ttl_budget_counts_initial_offset_and_speculative_native_page(setup, monkeypatch):
    from openviking.storage import viking_vector_index_backend as module
    from openviking_cli.exceptions import ResourceExhaustedError

    s = setup
    monkeypatch.setattr(module, "_TTL_MAX_CANDIDATES", 2)
    with pytest.raises(ResourceExhaustedError):
        await s.backend.query(ctx=s.ctx, limit=1, offset=2, include_expired=False)
    assert s.calls == []
    s.fs.ttl_registry.account_may_have_records = AsyncMock(side_effect=[False, True])
    with pytest.raises(ResourceExhaustedError):
        await s.backend.query(ctx=s.ctx, limit=1, offset=1, include_expired=False)
    assert [(call["limit"], call["offset"]) for call in s.calls] == [(1, 1)]


@pytest.mark.asyncio
async def test_native_page_can_exceed_ttl_budget_when_account_stays_unmanaged(setup, monkeypatch):
    from openviking.storage import viking_vector_index_backend as module

    s = setup
    monkeypatch.setattr(module, "_TTL_MAX_CANDIDATES", 1)
    s.fs.ttl_registry.account_may_have_records = AsyncMock(return_value=False)
    s.rows.extend({"uri": f"{ROOT}/2026/09/01/{i}.md"} for i in range(3))
    result = await s.backend.query(ctx=s.ctx, limit=2, include_expired=False)
    assert len(result) == 2
    assert len(s.calls) == 1
    s.fs._async_agfs.read.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["query", "search_by_keywords", "search_by_random"])
@pytest.mark.parametrize(
    "offset,limit", [(0, 10), (100, 10), (115, 10), (120, 10), (9000, 10), (0, 9000)]
)
async def test_unmanaged_pagination_matches_native_backend(setup, method, offset, limit):
    """TTL-off pages preserve native filtering, order, scores and page boundaries."""
    s = setup
    s.fs.ttl_registry.account_may_have_records = AsyncMock(return_value=False)
    # 120 matching rows interleaved with rows rejected by the backend filter.
    s.rows.extend(
        {
            "uri": f"{ROOT}/2026/09/01/{i}.md",
            "level": i % 2,
            "account_id": "acct",
            "_score": 240 - i,
        }
        for i in range(240)
    )
    predicate = Eq("level", 1)
    projection = ["_score", "level"]
    native = await getattr(s.single, method)(
        filter=predicate, limit=limit, offset=offset, output_fields=projection
    )
    s.calls.clear()
    kwargs = {
        "ctx": s.ctx,
        "filter": predicate,
        "limit": limit,
        "offset": offset,
        "output_fields": projection,
    }
    if method == "query":
        kwargs["include_expired"] = False
    elif method == "search_by_keywords":
        kwargs.update(query="meeting", mode="bm25", fields=["content"])
    result = await getattr(s.backend, method)(**kwargs)
    # The TTL response contract adds null expiry; native fields remain unchanged.
    assert result == [{**row, "expires_at": None} for row in native]
    assert len(s.calls) == 1
    assert (s.calls[0]["offset"], s.calls[0]["limit"]) == (offset, limit)
    s.fs._async_agfs.read.assert_not_awaited()
    if method == "search_by_keywords":
        assert s.calls[0]["query"] == "meeting"
        assert s.calls[0]["mode"] == "bm25"
        assert s.calls[0]["fields"] == ["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("descending", [False, True])
async def test_unmanaged_sorted_offset_matches_native_order(setup, descending):
    s = setup
    s.fs.ttl_registry.account_may_have_records = AsyncMock(return_value=False)
    s.rows.extend(
        {"uri": f"{ROOT}/2026/09/01/{i}.md", "rank": i, "_score": i / 120}
        for i in reversed(range(120))
    )
    advance = {"time_decay": {"protection": "0"}}
    result = await s.backend.query(
        ctx=s.ctx,
        limit=10,
        offset=100,
        order_by="rank",
        order_desc=descending,
        output_fields=["rank", "_score"],
        advance=advance,
        include_expired=False,
    )
    expected = sorted(s.rows, key=lambda row: row["rank"], reverse=descending)[100:110]
    assert result == [
        {"rank": row["rank"], "_score": row["_score"], "expires_at": None} for row in expected
    ]
    assert len(s.calls) == 1
    assert s.calls[0]["advance"] == advance
    assert (s.calls[0]["offset"], s.calls[0]["limit"]) == (100, 10)
    s.fs._async_agfs.read.assert_not_awaited()
