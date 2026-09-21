# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.acl import AclManager
from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.expr import And, Eq, In, Or, PathScope, RawDSL
from openviking.storage.viking_vector_index_backend import (
    VikingVectorIndexBackend,
    _SingleAccountBackend,
)
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.vectordb_config import VectorDBBackendConfig


def _ctx(*, role: Role = Role.USER, actor_peer_id: str | None = None) -> RequestContext:
    return RequestContext(
        user=UserIdentifier("acct", "alice"),
        role=role,
        actor_peer_id=actor_peer_id,
    )


def _build(
    ctx: RequestContext,
    targets: list[str] | None,
    *,
    context_type: str | None = "resource",
    extra_filter=None,
    level: list[int] | None = None,
    acl_enabled: bool = False,
):
    backend = object.__new__(VikingVectorIndexBackend)
    backend.acl_manager = None
    return backend._build_scope_filter(
        ctx=ctx,
        context_type=context_type,
        target_directories=targets,
        extra_filter=extra_filter,
        level=level,
        acl_enabled=acl_enabled,
    )


def _tenant_filter(ctx: RequestContext, *, acl_enabled: bool = False):
    backend = object.__new__(VikingVectorIndexBackend)
    backend.acl_manager = None
    return backend._tenant_filter(ctx, acl_enabled=acl_enabled)


class _AclConfigReader:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    async def get_account(self, account_id: str, field: str):
        del account_id, field
        return SimpleNamespace(enabled=self.enabled)


class _FailingAsyncAdapter:
    async def call(self, method_name, **kwargs):
        raise RuntimeError(f"{method_name} failed")


class _RecordingAsyncAdapter:
    def __init__(self):
        self.calls = []

    async def call(self, method_name, **kwargs):
        self.calls.append((method_name, kwargs))
        return []


@pytest.mark.asyncio
async def test_search_by_random_passes_runtime_acl_state_to_tenant_filter():
    ctx = _ctx()
    adapter = SimpleNamespace(search_by_random=AsyncMock(return_value=[]))
    acl_reader = _AclConfigReader(False)
    backend = object.__new__(VikingVectorIndexBackend)
    backend.acl_manager = AclManager(backend, acl_reader)
    backend._get_backend_for_context = lambda _ctx: adapter

    assert await backend.search_by_random(ctx=ctx) == []
    adapter.search_by_random.assert_awaited_once()
    assert adapter.search_by_random.await_args.kwargs["filter"] == _tenant_filter(ctx)


def test_descendant_target_elides_only_visible_root_path_filter():
    ctx = _ctx()
    target = "viking://resources/wiki/physics"

    result = _build(
        ctx,
        [target],
        extra_filter=Eq("status", "ready"),
        level=[2],
    )

    assert result == And(
        [
            Eq("context_type", "resource"),
            Eq("account_id", "acct"),
            Or([PathScope("uri", target, depth=-1)]),
            Eq("status", "ready"),
            In("level", [2]),
        ]
    )


def test_equal_visible_root_elides_only_visible_root_path_filter():
    ctx = _ctx()

    result = _build(ctx, ["viking://resources"])

    assert result == And(
        [
            Eq("context_type", "resource"),
            Eq("account_id", "acct"),
            Or([PathScope("uri", "viking://resources", depth=-1)]),
        ]
    )


def test_all_targets_may_be_under_different_visible_roots():
    ctx = _ctx()
    targets = [
        "viking://resources/wiki/physics",
        "viking://user/alice/resources/private-notes",
        "viking://agent/tools/search",
    ]

    result = _build(ctx, targets)

    assert result == And(
        [
            Eq("context_type", "resource"),
            Eq("account_id", "acct"),
            Or(
                [
                    PathScope("uri", "viking://resources/wiki/physics", depth=-1),
                    PathScope("uri", "viking://user/alice/resources/private-notes", depth=-1),
                    PathScope("uri", "viking://agent/tools/search", depth=-1),
                ]
            ),
        ]
    )


def test_mixed_visible_and_outside_targets_keep_original_tenant_filter():
    ctx = _ctx()
    targets = ["viking://resources/wiki", "viking://upload/staged"]

    result = _build(ctx, targets)

    assert result == And(
        [
            Eq("context_type", "resource"),
            _tenant_filter(ctx),
            Or([PathScope("uri", target, depth=-1) for target in targets]),
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_mode", [{}, {"acl_mode": None}, {"acl_mode": "none"}])
async def test_tenant_search_enforces_visible_roots_and_shared_acl(tmp_path, legacy_mode):
    ctx = _ctx()
    own_uri = "viking://user/alice/resources/notes"
    cross_user_uri = "viking://user/bob/resources/notes"
    records = [
        {
            "id": "own",
            "uri": own_uri,
            "account_id": "acct",
            "context_type": "resource",
        },
        {
            "id": "cross-user",
            "uri": cross_user_uri,
            "account_id": "acct",
            "context_type": "resource",
        },
        {
            **legacy_mode,
            "id": "legacy-shared",
            "uri": "viking://agent/workflows/daily.md",
            "account_id": "acct",
            "context_type": "resource",
        },
        {
            "id": "direct-shared",
            "uri": "viking://resources/direct.md",
            "account_id": "acct",
            "context_type": "resource",
            "acl_mode": "inherit",
            "acl_direct_grants": ["1:user:alice"],
        },
        {
            "id": "inherited-shared",
            "uri": "viking://resources/inherited.md",
            "account_id": "acct",
            "context_type": "resource",
            "acl_mode": "inherit",
            "acl_inherited_grants": ["3:user:*"],
        },
        {
            "id": "restricted-inherited-shared",
            "uri": "viking://resources/restricted-inherited.md",
            "account_id": "acct",
            "context_type": "resource",
            "acl_mode": "restricted",
            "acl_inherited_grants": ["3:user:*"],
        },
        {
            "id": "restricted-direct-shared",
            "uri": "viking://resources/restricted-direct.md",
            "account_id": "acct",
            "context_type": "resource",
            "acl_mode": "restricted",
            "acl_direct_grants": ["1:user:alice"],
            "acl_inherited_grants": ["7:user:bob"],
        },
        {
            "id": "denied-shared",
            "uri": "viking://resources/denied.md",
            "account_id": "acct",
            "context_type": "resource",
            "acl_mode": "inherit",
            "acl_direct_grants": ["7:user:bob"],
        },
        {
            "id": "foreign-account",
            "uri": "viking://resources/foreign.md",
            "account_id": "other",
            "context_type": "resource",
        },
    ]

    backend = VikingVectorIndexBackend(
        config=VectorDBBackendConfig(
            backend="local", name="context", dimension=4, path=str(tmp_path / "vectors")
        )
    )
    try:
        schema = CollectionSchemas.context_collection("context", 4)
        # Exercise genuinely absent/null fields without a schema default filling them in.
        next(field for field in schema["Fields"] if field["FieldName"] == "acl_mode").pop(
            "DefaultValue"
        )
        assert await backend.create_collection("context", schema)
        acl_config = _AclConfigReader(True)
        backend.acl_manager = AclManager(backend, acl_config)
        for record in records:
            record_ctx = RequestContext(
                user=UserIdentifier(record["account_id"], ctx.user.user_id), role=Role.ADMIN
            )
            await backend._upsert_many_raw(
                [{**record, "level": 2, "vector": [1.0, 0.0, 0.0, 0.0]}], ctx=record_ctx
            )

        visible = await backend.search_in_tenant(
            ctx=ctx,
            query_vector=[1.0, 0.0, 0.0, 0.0],
            context_type="resource",
        )
        cross_user_only = await backend.search_in_tenant(
            ctx=ctx,
            query_vector=[1.0, 0.0, 0.0, 0.0],
            context_type="resource",
            target_directories=[cross_user_uri],
        )
        internal = await backend.search_in_tenant(
            ctx=RequestContext(
                user=ctx.user,
                role=ctx.role,
                bypass_acl=True,
            ),
            query_vector=[1.0, 0.0, 0.0, 0.0],
            context_type="resource",
        )

        assert sorted(record["id"] for record in visible) == sorted(
            [
                "own",
                "legacy-shared",
                "direct-shared",
                "inherited-shared",
                "restricted-direct-shared",
            ]
        )
        assert cross_user_only == []
        assert sorted(record["id"] for record in internal) == sorted(
            [
                "own",
                "cross-user",
                "legacy-shared",
                "direct-shared",
                "inherited-shared",
                "restricted-inherited-shared",
                "restricted-direct-shared",
                "denied-shared",
            ]
        )

        acl_config.enabled = False
        shared = await backend.search_in_tenant(
            ctx=ctx,
            query_vector=[1.0, 0.0, 0.0, 0.0],
            context_type="resource",
        )
        assert sorted(record["id"] for record in shared) == sorted(
            [
                "own",
                "legacy-shared",
                "direct-shared",
                "inherited-shared",
                "restricted-inherited-shared",
                "restricted-direct-shared",
                "denied-shared",
            ]
        )

    finally:
        await backend.close()


def test_segment_prefix_and_visible_root_ancestor_do_not_elide_tenant_filter():
    ctx = _ctx()

    segment_prefix = _build(ctx, ["viking://resources-other/wiki"])
    ancestor = _build(ctx, ["viking://user"])

    assert segment_prefix == And(
        [
            Eq("context_type", "resource"),
            _tenant_filter(ctx),
            Or([PathScope("uri", "viking://resources-other/wiki", depth=-1)]),
        ]
    )
    assert ancestor == And(
        [
            Eq("context_type", "resource"),
            _tenant_filter(ctx),
            Or([PathScope("uri", "viking://user", depth=-1)]),
        ]
    )


def test_no_target_keeps_original_tenant_filter():
    ctx = _ctx()

    assert _build(ctx, None) == And(
        [
            Eq("context_type", "resource"),
            _tenant_filter(ctx),
        ]
    )


def test_merge_filters_wraps_raw_dict_filter():
    backend = object.__new__(VikingVectorIndexBackend)

    result = backend._merge_filters(
        {"op": "must", "field": "uri", "conds": ["viking://resources"]},
        Eq("account_id", "acct"),
    )

    assert result == And(
        [
            RawDSL({"op": "must", "field": "uri", "conds": ["viking://resources"]}),
            Eq("account_id", "acct"),
        ]
    )


def test_root_role_keeps_existing_target_only_behavior():
    ctx = _ctx(role=Role.ROOT)
    target = "viking://resources/wiki"

    assert _build(ctx, [target]) == And(
        [
            Eq("context_type", "resource"),
            Or([PathScope("uri", target, depth=-1)]),
        ]
    )


def test_actor_peer_target_retains_account_and_exact_target_scope():
    ctx = _ctx(actor_peer_id="visitor-a")
    target = "viking://user/alice/peers/visitor-a/resources/cases"

    result = _build(ctx, [target])

    assert result == And(
        [
            Eq("context_type", "resource"),
            Eq("account_id", "acct"),
            Or([PathScope("uri", target, depth=-1)]),
        ]
    )


@pytest.mark.asyncio
async def test_search_by_random_propagates_adapter_errors():
    backend = object.__new__(_SingleAccountBackend)
    backend._bound_account_id = None
    backend._async_adapter = _FailingAsyncAdapter()

    with pytest.raises(RuntimeError, match="search_by_random failed"):
        await backend.search_by_random(filter=Eq("uri", "viking://resources/a.md"))


@pytest.mark.asyncio
async def test_search_by_random_reuses_account_filter_for_raw_dsl():
    backend = object.__new__(_SingleAccountBackend)
    backend._bound_account_id = "acct"
    backend._async_adapter = _RecordingAsyncAdapter()
    raw_filter = {"op": "must", "field": "uri", "conds": ["viking://resources"]}

    await backend.search_by_random(filter=raw_filter)

    assert backend._async_adapter.calls == [
        (
            "search_by_random",
            {
                "filter": And([Eq("account_id", "acct"), RawDSL(raw_filter)]),
                "limit": 10,
                "offset": 0,
                "output_fields": None,
                "advance": None,
            },
        )
    ]


@pytest.mark.asyncio
async def test_zero_decay_weight_keeps_the_original_single_search_call():
    backend = object.__new__(VikingVectorIndexBackend)
    backend.acl_manager = None
    calls = []

    async def fake_search(**kwargs):
        calls.append(kwargs)
        return [{"uri": "viking://resources/doc", "_score": 0.8}]

    backend.search = fake_search
    results = await backend.search_in_tenant(
        ctx=_ctx(),
        query_vector=[1.0],
        context_type="memory",
        limit=7,
        offset=2,
        events_time_decay_weight=0.0,
    )

    assert results == [{"uri": "viking://resources/doc", "_score": 0.8}]
    assert len(calls) == 1
    assert calls[0]["limit"] == 7
    assert calls[0]["offset"] == 2


@pytest.mark.asyncio
async def test_decay_fuses_user_and_peer_events_from_one_candidate_search():
    backend = object.__new__(VikingVectorIndexBackend)
    backend.acl_manager = None
    calls = []

    async def fake_search(**kwargs):
        calls.append(kwargs)
        return [
            {
                "uri": "viking://user/alice/memories/events/old",
                "level": 2,
                "updated_at": "2026-01-01T00:00:00Z",
                "_score": 0.9,
            },
            {
                "uri": "viking://user/alice/peers/peer-a/memories/events/recent",
                "level": 2,
                "updated_at": "2026-01-08T00:00:00Z",
                "_score": 0.6,
            },
            {
                "uri": "viking://user/alice/memories/preferences/pref.md",
                "level": 2,
                "updated_at": "2026-01-08T00:00:00Z",
                "_score": 0.75,
            },
        ]

    backend.search = fake_search
    results = await backend.search_in_tenant(
        ctx=_ctx(),
        query_vector=[1.0],
        context_type="memory",
        limit=3,
        events_time_decay_weight=0.5,
        request_now=datetime(2026, 1, 8, tzinfo=timezone.utc),
    )

    assert len(calls) == 1
    assert calls[0]["limit"] == 9
    assert [item["uri"] for item in results] == [
        "viking://user/alice/peers/peer-a/memories/events/recent",
        "viking://user/alice/memories/preferences/pref.md",
        "viking://user/alice/memories/events/old",
    ]
    assert results[0]["_origin_score"] == pytest.approx(0.6)
    assert results[0]["_time_score"] == pytest.approx(1.0)
    assert "_origin_score" not in results[1]
    assert "_time_score" not in results[1]
    assert results[2]["_time_score"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_decay_rerank_prefetch_keeps_expanded_origin_candidates():
    backend = object.__new__(VikingVectorIndexBackend)
    backend.acl_manager = None
    calls = []

    async def fake_search(**kwargs):
        calls.append(kwargs)
        candidates = [
            {
                "uri": "viking://user/alice/memories/events/fresh",
                "level": 2,
                "_score": 0.9,
                "updated_at": "2026-01-08T00:00:00Z",
            },
            {
                "uri": "viking://user/alice/peers/peer-a/memories/events/old",
                "level": 2,
                "_score": 0.8,
                "updated_at": "2026-01-01T00:00:00Z",
            },
        ]
        return candidates[: kwargs["limit"]]

    backend.search = fake_search
    results = await backend.search_in_tenant(
        ctx=_ctx(),
        query_vector=[1.0],
        context_type="memory",
        target_directories=["viking://user/alice"],
        level=[2],
        limit=1,
        events_time_decay_weight=0.8,
        for_rerank=True,
        request_now=datetime(2026, 1, 8, tzinfo=timezone.utc),
    )

    assert calls[0]["limit"] == 3
    assert [result["_score"] for result in results] == pytest.approx([0.9, 0.8])
    assert [result["_origin_score"] for result in results] == pytest.approx([0.9, 0.8])
    assert [result["_time_score"] for result in results] == pytest.approx([1.0, 0.5])


@pytest.mark.asyncio
async def test_decay_applies_to_a_peer_only_target():
    backend = object.__new__(VikingVectorIndexBackend)
    backend.acl_manager = None
    calls = []

    async def fake_search(**kwargs):
        calls.append(kwargs)
        return [
            {
                "uri": "viking://user/alice/peers/peer-a/memories/events/recent",
                "level": 2,
                "updated_at": "2026-01-08T00:00:00Z",
                "_score": 0.2,
            }
        ]

    backend.search = fake_search
    results = await backend.search_in_tenant(
        ctx=_ctx(),
        query_vector=[1.0],
        context_type="memory",
        target_directories=["viking://user/alice/peers/peer-a/memories/events"],
        level=[2],
        events_time_decay_weight=0.25,
        request_now=datetime(2026, 1, 8, tzinfo=timezone.utc),
    )

    assert len(calls) == 1
    assert calls[0]["limit"] == 30
    assert results[0]["_score"] == pytest.approx(0.4)
    assert results[0]["_origin_score"] == pytest.approx(0.2)
    assert results[0]["_time_score"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_decay_applies_under_bare_user_target():
    backend = object.__new__(VikingVectorIndexBackend)
    backend.acl_manager = None
    calls = []

    async def fake_search(**kwargs):
        calls.append(kwargs)
        return []

    backend.search = fake_search
    await backend.search_in_tenant(
        ctx=_ctx(),
        query_vector=[1.0],
        context_type="memory",
        target_directories=["viking://user"],
        level=[2],
        limit=2,
        events_time_decay_weight=0.25,
    )

    assert len(calls) == 1
    assert calls[0]["limit"] == 6


@pytest.mark.asyncio
async def test_decay_applies_to_children_of_a_peer_event_directory():
    backend = object.__new__(VikingVectorIndexBackend)
    backend.acl_manager = None
    calls = []

    async def fake_search(**kwargs):
        calls.append(kwargs)
        return [
            {
                "uri": "viking://user/alice/peers/peer-a/memories/events/recent",
                "level": 2,
                "updated_at": "2026-01-08T00:00:00Z",
                "_score": 0.2,
            }
        ]

    backend.search = fake_search
    results = await backend.search_children_in_tenant(
        ctx=_ctx(),
        parent_uri="viking://user/alice/peers/peer-a/memories/events",
        query_vector=[1.0],
        context_type="memory",
        limit=2,
        events_time_decay_weight=0.25,
        request_now=datetime(2026, 1, 8, tzinfo=timezone.utc),
    )

    assert len(calls) == 1
    assert calls[0]["limit"] == 6
    assert results[0]["_score"] == pytest.approx(0.2)
    assert results[0]["_origin_score"] == pytest.approx(0.2)
    assert results[0]["_time_score"] == pytest.approx(1.0)
