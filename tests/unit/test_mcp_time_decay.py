# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from types import SimpleNamespace

import openviking.server.mcp_endpoint as mcp_endpoint
from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier


async def test_find_exposes_and_forwards_event_time_decay(monkeypatch):
    captured = {}

    async def fake_find(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(memories=[], resources=[], skills=[])

    service = SimpleNamespace(search=SimpleNamespace(find=fake_find))
    monkeypatch.setattr(mcp_endpoint, "get_service", lambda: service)
    token = mcp_endpoint._mcp_ctx.set(
        RequestContext(
            user=UserIdentifier.the_default_user("test_user"),
            role=Role.ROOT,
        )
    )
    try:
        result = await mcp_endpoint.find(
            query="recent decision",
            context_type="memory",
            events_time_decay_protection="7d",
        )
        tools = {tool.name: tool for tool in await mcp_endpoint.mcp.list_tools()}
    finally:
        mcp_endpoint._mcp_ctx.reset(token)

    assert result == "No matching context found."
    assert captured["events_time_decay_protection"] == "7d"
    assert "events_time_decay_protection" in tools["find"].inputSchema["properties"]
