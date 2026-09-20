# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Unit tests for search time-decay request validation and routing."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from openviking.server.identity import RequestContext, Role
from openviking.server.routers import search as search_router
from openviking_cli.session.user_id import UserIdentifier


def _request_context() -> RequestContext:
    return RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)


async def test_search_router_forwards_time_decay_parameters(monkeypatch):
    captured = {}

    async def fake_search(**kwargs):
        captured.update(kwargs)
        return {"items": []}

    monkeypatch.setattr(
        search_router,
        "get_service",
        lambda: SimpleNamespace(
            search=SimpleNamespace(search=fake_search),
            sessions=SimpleNamespace(),
        ),
    )

    response = await search_router.search(
        search_router.SearchRequest(
            query="sample",
            events_time_decay_weight=0.25,
            events_time_decay_protection="2d",
        ),
        _request_context(),
    )

    assert response["status"] == "ok"
    assert captured["events_time_decay_weight"] == 0.25
    assert captured["events_time_decay_protection"] == "2d"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("events_time_decay_weight", -0.1),
        ("events_time_decay_weight", 1.0),
        ("events_time_decay_weight", float("nan")),
        ("events_time_decay_weight", float("inf")),
        ("events_time_decay_weight", False),
        ("events_time_decay_weight", True),
        ("events_time_decay_protection", "1w"),
        ("events_time_decay_protection", "-1d"),
    ],
)
def test_search_request_rejects_invalid_time_decay_parameters(field, value):
    with pytest.raises(ValidationError):
        search_router.SearchRequest(query="sample", **{field: value})


def test_search_request_keeps_numeric_string_weight_compatibility():
    request = search_router.SearchRequest(
        query="sample",
        events_time_decay_weight="0.25",
    )

    assert request.events_time_decay_weight == 0.25


def test_find_request_rejects_time_decay_parameters():
    with pytest.raises(ValidationError):
        search_router.FindRequest(query="sample", events_time_decay_weight=0.25)


def test_context_mode_rejects_time_decay_parameters():
    with pytest.raises(ValidationError, match="only supported in mode='list'"):
        search_router.SearchRequest(
            query="sample",
            mode="context",
            events_time_decay_weight=0.25,
        )
