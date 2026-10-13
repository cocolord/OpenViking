# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import asyncio
import contextvars
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.viking_fs import VikingFS
from openviking.telemetry import OperationTelemetry, bind_telemetry
from openviking_cli.retrieve.types import ContextType, MatchedContext, QueryResult
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.retrieval_config import RetrievalConfig


def _ctx() -> RequestContext:
    return RequestContext(user=UserIdentifier("acc1", "user1"), role=Role.USER)


def _make_viking_fs(*, enable_intent: bool) -> VikingFS:
    fs = VikingFS.__new__(VikingFS)
    fs.agfs = MagicMock()
    fs.acl_manager = None
    fs.query_embedder = MagicMock(name="embedder")
    fs.rerank_config = None
    fs.retrieval_config = RetrievalConfig(enable_intent=enable_intent)
    fs.vector_store = MagicMock(name="vector_store")
    fs._bound_ctx = contextvars.ContextVar("vikingfs_bound_ctx_intent_test", default=None)
    fs._ensure_access = AsyncMock()
    fs._get_vector_store = MagicMock(return_value=fs.vector_store)
    fs._get_embedder = MagicMock(return_value=fs.query_embedder)
    fs._ctx_or_default = MagicMock(return_value=_ctx())
    fs.abstract = AsyncMock(return_value="")
    return fs


@pytest.fixture
def intent_search(monkeypatch):
    fs = _make_viking_fs(enable_intent=True)
    planner = SimpleNamespace(
        get_completion_async=AsyncMock(
            return_value=json.dumps(
                {"queries": [{"query": "planned query", "context_type": "resource"}]}
            )
        )
    )
    fs._vlm_resolver = SimpleNamespace(get_query_planner=AsyncMock(return_value=planner))
    match = MatchedContext(
        uri="viking://resources/docs/answer.md", context_type=ContextType.RESOURCE
    )

    async def retrieve(typed_query, **kwargs):
        return QueryResult(
            query=typed_query,
            matched_contexts=[match],
            searched_directories=typed_query.target_directories,
        )

    retriever = SimpleNamespace(retrieve=AsyncMock(side_effect=retrieve))
    monkeypatch.setattr(
        "openviking.retrieve.hierarchical_retriever.HierarchicalRetriever",
        MagicMock(return_value=retriever),
    )
    return SimpleNamespace(fs=fs, planner=planner, retriever=retriever, match=match)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "session_info",
    [
        {"latest_archive_overview": "previous discussion"},
        {"current_messages": [SimpleNamespace(role="user", content="previous turn")]},
    ],
)
async def test_search_intent_timeout_cancels_planner_and_retrieves_raw_query(
    intent_search, session_info
):
    fs = intent_search.fs
    fs.retrieval_config.recall_intent_timeout_s = 0.02
    cancelled = asyncio.Event()

    async def blocked_planner(prompt):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    intent_search.planner.get_completion_async.side_effect = blocked_planner
    scope_filter = {"category": "docs"}
    telemetry = OperationTelemetry(operation="search.search", enabled=True)
    with bind_telemetry(telemetry):
        result = await asyncio.wait_for(
            fs.search(
                "raw query",
                target_uri="viking://resources/docs",
                session_info=session_info,
                ctx=_ctx(),
                limit=3,
                score_threshold=0.4,
                filter=scope_filter,
                level=[2],
            ),
            timeout=1.0,
        )

    assert cancelled.is_set()
    assert result.query_plan is None
    assert result.resources == [intent_search.match]
    intent_search.planner.get_completion_async.assert_awaited_once()
    intent_search.retriever.retrieve.assert_awaited_once()
    args = intent_search.retriever.retrieve.await_args
    assert args.args[0].query == "raw query"
    assert args.args[0].target_directories == ["viking://resources/docs"]
    assert args.kwargs["ctx"].account_id == "acc1"
    assert args.kwargs["limit"] == 3
    assert args.kwargs["score_threshold"] == 0.4
    assert args.kwargs["scope_dsl"] is scope_filter
    assert args.kwargs["level"] == [2]
    snapshot = telemetry.finish()
    assert snapshot is not None
    summary = snapshot.summary
    assert summary["search"]["typed_queries_count"] == 1
    assert summary["search"]["intent_analysis"]["status"] == "failed"
    assert summary["search"]["intent_analysis"]["duration_ms"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["provider", "malformed_response", "resolver"])
async def test_search_intent_failure_retrieves_raw_query(intent_search, failure):
    if failure == "provider":
        intent_search.planner.get_completion_async.side_effect = RuntimeError("provider overloaded")
    elif failure == "malformed_response":
        intent_search.planner.get_completion_async.return_value = "not a query plan"
    else:
        intent_search.fs._vlm_resolver.get_query_planner.side_effect = RuntimeError(
            "planner unavailable"
        )

    result = await intent_search.fs.search(
        "raw query",
        target_uri="viking://resources/docs",
        session_info={"latest_archive_overview": "previous discussion"},
        ctx=_ctx(),
    )

    assert result.query_plan is None
    assert result.resources == [intent_search.match]
    intent_search.retriever.retrieve.assert_awaited_once()
    typed_query = intent_search.retriever.retrieve.await_args.args[0]
    assert typed_query.query == "raw query"
    assert typed_query.target_directories == ["viking://resources/docs"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        ("timeout", "failed"),
        ("provider", "failed"),
        ("disabled", "off"),
        ("empty_session", "off"),
        (None, "used"),
    ],
)
async def test_search_endpoint_with_session_reports_intent_status(
    intent_search, monkeypatch, failure, expected_status
):
    import httpx
    from fastapi import FastAPI

    from openviking.server.auth import get_request_context
    from openviking.server.routers import search as search_router
    from openviking.service.search_service import SearchService

    if failure == "timeout":
        intent_search.fs.retrieval_config.recall_intent_timeout_s = 0.02

        async def blocked_planner(prompt):
            await asyncio.Event().wait()

        intent_search.planner.get_completion_async.side_effect = blocked_planner
    elif failure == "provider":
        intent_search.planner.get_completion_async.side_effect = RuntimeError("provider overloaded")
    elif failure == "disabled":
        intent_search.fs.retrieval_config.enable_intent = False

    session = SimpleNamespace(
        load=AsyncMock(),
        get_context_for_search=AsyncMock(
            return_value={}
            if failure == "empty_session"
            else {"latest_archive_overview": "previous discussion"}
        ),
    )
    service = SimpleNamespace(
        search=SearchService(intent_search.fs),
        sessions=SimpleNamespace(session=MagicMock(return_value=session)),
    )
    monkeypatch.setattr(search_router, "get_service", lambda: service)
    app = FastAPI()
    app.include_router(search_router.router)
    app.dependency_overrides[get_request_context] = _ctx
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await asyncio.wait_for(
            client.post(
                "/api/v1/search/search",
                json={
                    "query": "raw query",
                    "session_id": "session-with-context",
                    "target_uri": "viking://resources/docs",
                    "context_type": "resource",
                    "telemetry": True,
                },
            ),
            timeout=1.0,
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["result"]["total"] == 1
    assert body["result"]["resources"][0]["uri"] == intent_search.match.uri
    if expected_status == "used":
        assert body["result"]["query_plan"]["queries"][0]["query"] == "planned query"
    else:
        assert "query_plan" not in body["result"]
    assert body["telemetry"]["summary"]["search"]["typed_queries_count"] == 1
    assert body["telemetry"]["summary"]["search"]["intent_analysis"]["status"] == expected_status
    if failure == "disabled":
        session.load.assert_not_awaited()
        session.get_context_for_search.assert_not_awaited()
    else:
        session.load.assert_awaited_once()
        session.get_context_for_search.assert_awaited_once_with("raw query")
    args = intent_search.retriever.retrieve.await_args
    assert args.args[0].query == ("planned query" if expected_status == "used" else "raw query")
    assert args.args[0].target_directories == ["viking://resources/docs"]
    assert args.kwargs["scope_dsl"] == {
        "op": "must",
        "field": "context_type",
        "conds": ["resource"],
    }


@pytest.mark.asyncio
async def test_search_cancellation_propagates_without_raw_query_fallback(intent_search):
    started = asyncio.Event()

    async def blocked_planner(prompt):
        started.set()
        await asyncio.Event().wait()

    intent_search.planner.get_completion_async.side_effect = blocked_planner
    task = asyncio.create_task(
        intent_search.fs.search(
            "raw query", session_info={"latest_archive_overview": "discussion"}, ctx=_ctx()
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    intent_search.retriever.retrieve.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_preserves_successful_empty_query_plan(intent_search):
    intent_search.planner.get_completion_async.return_value = '{"queries": []}'

    result = await intent_search.fs.search(
        "raw query", session_info={"latest_archive_overview": "discussion"}, ctx=_ctx()
    )

    assert result.query_plan is not None
    assert result.query_plan.queries == []
    assert result.total == 0
    intent_search.retriever.retrieve.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_retrieval_failure_propagates_after_intent_fallback(intent_search):
    intent_search.planner.get_completion_async.side_effect = RuntimeError("provider overloaded")
    intent_search.retriever.retrieve.side_effect = RuntimeError("storage unavailable")

    with pytest.raises(RuntimeError, match="storage unavailable"):
        await intent_search.fs.search(
            "raw query", session_info={"latest_archive_overview": "discussion"}, ctx=_ctx()
        )


def test_retrieval_config_enable_intent_defaults_true():
    cfg = RetrievalConfig()
    assert cfg.enable_intent is True


def test_retrieval_config_enable_intent_can_disable():
    cfg = RetrievalConfig(enable_intent=False)
    assert cfg.enable_intent is False


@pytest.mark.asyncio
async def test_search_skips_intent_and_uses_raw_query_when_disabled(monkeypatch):
    fs = _make_viking_fs(enable_intent=False)
    captured = {}

    class ForbiddenIntentAnalyzer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("intent analysis must not run when disabled")

    class FakeRetriever:
        def __init__(self, storage, embedder, rerank_config):
            pass

        async def retrieve(self, typed_query, **kwargs):
            captured["typed_query"] = typed_query
            return QueryResult(
                query=typed_query,
                matched_contexts=[],
                searched_directories=typed_query.target_directories,
            )

    monkeypatch.setattr(
        "openviking.retrieve.intent_analyzer.IntentAnalyzer",
        ForbiddenIntentAnalyzer,
    )
    monkeypatch.setattr(
        "openviking.retrieve.hierarchical_retriever.HierarchicalRetriever",
        FakeRetriever,
    )

    result = await fs.search(
        "raw query",
        target_uri="viking://resources/docs",
        session_info={
            "latest_archive_overview": "previous summary",
            "current_messages": [{"role": "user", "content": "previous turn"}],
        },
        ctx=_ctx(),
    )

    assert result.query_plan is None
    assert captured["typed_query"].query == "raw query"
    assert captured["typed_query"].intent == ""
    assert captured["typed_query"].target_directories == ["viking://resources/docs"]


def test_search_service_is_intent_enabled_follows_config():
    from openviking.service.search_service import SearchService

    svc = SearchService(_make_viking_fs(enable_intent=False))
    assert svc.is_intent_enabled() is False

    svc.set_viking_fs(_make_viking_fs(enable_intent=True))
    assert svc.is_intent_enabled() is True

    empty = SearchService()
    assert empty.is_intent_enabled() is True


@pytest.mark.asyncio
async def test_search_service_skips_session_context_when_intent_disabled():
    from openviking.service.search_service import SearchService

    fs = _make_viking_fs(enable_intent=False)
    fs.search = AsyncMock(return_value=MagicMock(name="find_result", query_plan=None, total=0))
    session = MagicMock()
    session.get_context_for_search = AsyncMock(
        side_effect=AssertionError("must not scan session when intent disabled")
    )

    svc = SearchService(fs)
    await svc.search(query="hello", ctx=_ctx(), session=session, target_uri="")

    session.get_context_for_search.assert_not_awaited()
    assert fs.search.await_args.kwargs.get("session_info") is None


@pytest.mark.asyncio
async def test_search_service_loads_session_context_when_intent_enabled():
    from openviking.service.search_service import SearchService

    fs = _make_viking_fs(enable_intent=True)
    fs.search = AsyncMock(return_value=MagicMock(name="find_result", query_plan=None, total=0))
    session_info = {"latest_archive_overview": "ov", "current_messages": []}
    session = MagicMock()
    session.get_context_for_search = AsyncMock(return_value=session_info)

    svc = SearchService(fs)
    await svc.search(query="hello", ctx=_ctx(), session=session, target_uri="")

    session.get_context_for_search.assert_awaited_once()
    assert fs.search.await_args.kwargs.get("session_info") is session_info


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "planned", "expected"),
    [
        ("memory", "resource", "memory"),
        (["memory"], "skill", "memory"),
        ("resource", "memory", "resource"),
        (["memory", "resource"], "memory", "memory"),
        (["memory", "resource"], "skill", None),
        (None, "skill", "skill"),
    ],
)
async def test_session_search_plan_respects_explicit_context_scope(
    monkeypatch, scope, planned, expected
):
    # #5544: planner classification must not contradict the caller's REST scope.
    from starlette.requests import Request

    from openviking.server.routers import search as search_router
    from openviking.service.search_service import SearchService

    fs = _make_viking_fs(enable_intent=True)
    planner = SimpleNamespace(
        get_completion_async=AsyncMock(
            return_value=json.dumps(
                {"queries": [{"query": "known record", "context_type": planned}]}
            )
        )
    )
    fs._vlm_resolver = SimpleNamespace(get_query_planner=AsyncMock(return_value=planner))
    session = SimpleNamespace(
        load=AsyncMock(),
        get_context_for_search=AsyncMock(
            return_value={"latest_archive_overview": "previous discussion"}
        ),
    )
    service = SimpleNamespace(
        search=SearchService(fs),
        sessions=SimpleNamespace(session=lambda *args: session),
    )

    class FakeRetriever:
        def __init__(self, **kwargs):
            pass

        async def retrieve(self, typed_query, **kwargs):
            return QueryResult(query=typed_query, matched_contexts=[], searched_directories=[])

    monkeypatch.setattr(search_router, "get_service", lambda: service)
    monkeypatch.setattr(
        "openviking.retrieve.hierarchical_retriever.HierarchicalRetriever", FakeRetriever
    )
    response = await search_router.search(
        search_router.SearchRequest(query="known record", session_id="fixture", context_type=scope),
        Request({"type": "http"}),
        _ctx(),
    )

    assert response["status"] == "ok"
    assert response["result"]["query_plan"]["queries"][0]["context_type"] == expected
