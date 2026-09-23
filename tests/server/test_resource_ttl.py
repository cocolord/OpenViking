# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Public resource TTL through the real import pipeline and runtime policy store."""

import asyncio
import json
from datetime import timedelta

import pytest

from openviking.service.task_tracker import get_task_tracker
from openviking.storage.resource_ttl import prepare_resource_ttl
from openviking.utils.time_utils import format_iso8601, parse_iso_datetime
from openviking_cli.utils.config.open_viking_config import (
    get_openviking_config,
    set_openviking_config,
)
from tests.storage.test_transfer_merge_binding import root_ctx

ROOT = "viking://user/default/resources"


@pytest.fixture(autouse=True)
def restore_cluster_config():
    # Runtime PATCH publishes a process-wide singleton; keep later tests isolated.
    original = get_openviking_config()
    yield
    set_openviking_config(original)


@pytest.mark.asyncio
@pytest.mark.parametrize("parse_mode", ["default", "no_split"])
@pytest.mark.parametrize("root", [ROOT, "viking://resources"])
async def test_import_freezes_ttl_and_reimport_preserves_it(
    client, service, upload_temp_dir, parse_mode, root
):
    source = upload_temp_dir / "ttl-resource.md"
    source.write_text("# Guide\n\nAn imported resource with a frozen lifetime.\n")
    await service.viking_fs.mkdir(root, exist_ok=True, ctx=root_ctx())
    request = {
        "temp_file_id": source.name,
        **({"parent": root} if parse_mode == "no_split" else {"to": root + "/ttl-resource"}),
        "ttl_relative": 7,
        "wait": True,
        "args": {"parse_mode": parse_mode},
    }
    if root == "viking://resources":
        await service.runtime_config_manager.patch_account("default", {"acl": {"enabled": True}})
        request["acl"] = {
            "acl_mode": "restricted",
            "entries": [
                {"principal": "user:default", "level": "manage"},
                {"principal": "user:reader", "level": "read"},
            ],
        }
    response = await asyncio.wait_for(client.post("/api/v1/resources", json=request), 25)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    uri = result["root_uri"]
    if "acl" in request:
        acl = await service.viking_fs.get_acl(uri, ctx=root_ctx())
        assert acl["acl_mode"] == "restricted"
        assert acl["direct_entries"] == request["acl"]["entries"]
    response = await client.get("/api/v1/resources/ttl", params={"uri": uri})
    assert response.status_code == 200, response.text
    frozen = response.json()["result"]
    assert parse_iso_datetime(frozen["expires_at"]) - parse_iso_datetime(
        frozen["received_at"]
    ) == timedelta(days=7)
    record = await service.viking_fs.ttl_registry.get("default", uri)
    assert record.object_type == ("resource_file" if parse_mode == "no_split" else "resource")
    expiry = format_iso8601(parse_iso_datetime(frozen["expires_at"]) + timedelta(days=1))
    response = await client.patch("/api/v1/resources/ttl", json={"uri": uri, "expires_at": expiry})
    assert response.status_code == 200, response.text
    frozen["expires_at"] = expiry
    assert response.json()["result"] == frozen
    assert (await service.viking_fs.ttl_registry.get("default", uri)).expires_at == expiry
    source.write_text("# Guide\n\nUpdated resource body.\n")
    request["ttl_relative"] = 30
    if parse_mode == "no_split":
        result = await asyncio.wait_for(
            service.resources.refresh_resource(
                path=str(source),
                to=uri,
                to_is_directory=False,
                ctx=root_ctx(),
                ttl_relative=30,
                args={"parse_mode": "no_split"},
            ),
            25,
        )
        assert result["root_uri"] == uri
    else:
        response = await asyncio.wait_for(client.post("/api/v1/resources", json=request), 25)
        assert response.status_code == 200, response.text
    response = await client.get("/api/v1/resources/ttl", params={"uri": uri})
    assert response.json()["result"] == frozen


@pytest.mark.asyncio
@pytest.mark.parametrize("root", [ROOT, "viking://resources"])
async def test_directory_policy_is_incremental_and_keeps_cluster_defaults(client, service, root):
    fs, ctx = service.viking_fs, root_ctx()
    manager = service._runtime_config_manager
    await manager.patch_cluster({"ttl": {"global_default": {"mode": "days", "ttl_days": 30}}})
    policy_uri = root + "/policy"
    response = await client.patch(
        "/api/v1/resources/config", json={"uri": policy_uri, "ttl_relative": 7}
    )
    assert response.status_code == 200, response.text
    first = await prepare_resource_ttl(
        fs, policy_uri + "/one", is_dir=True, existing=False, ctx=ctx, lease_ref=None
    )
    outside = await prepare_resource_ttl(
        fs, ROOT + "/outside", is_dir=True, existing=False, ctx=ctx, lease_ref=None
    )
    assert first["ttl_days"] == 7
    assert outside["ttl_days"] == 30
    response = await client.patch(
        "/api/v1/resources/config", json={"uri": policy_uri, "ttl_relative": 14}
    )
    assert response.status_code == 200, response.text
    unchanged = await prepare_resource_ttl(
        fs, policy_uri + "/one", is_dir=True, existing=True, ctx=ctx, lease_ref=None
    )
    new = await prepare_resource_ttl(
        fs, policy_uri + "/two", is_dir=True, existing=False, ctx=ctx, lease_ref=None
    )
    assert unchanged == first
    assert new["ttl_days"] == 14
    response = await client.patch("/api/v1/resources/config", json={"uri": policy_uri})
    assert response.status_code == 200, response.text
    disabled = await prepare_resource_ttl(
        fs, policy_uri + "/three", is_dir=True, existing=False, ctx=ctx, lease_ref=None
    )
    assert disabled == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ttl",
    [{"ttl_relative": 0}, {"ttl_relative": True}, {"ttl_relative": 1, "ttl_absolute": 2000000000}],
)
async def test_api_rejects_invalid_ttl_before_import(client, ttl):
    response = await client.post(
        "/api/v1/resources", json={"path": "https://example.com/a.md", **ttl}
    )
    assert response.status_code == 400, response.text


@pytest.mark.asyncio
async def test_parent_watch_cannot_restore_independently_expired_child(service, upload_temp_dir):
    fs, ctx = service.viking_fs, root_ctx()
    source = upload_temp_dir / "source"
    source.mkdir()
    (source / "a.md").write_text("# Resource\n\nSource retained outside OpenViking.")
    root = ROOT + "/watched"
    await fs.mkdir(ROOT, exist_ok=True, ctx=ctx)
    await service.resources.add_resource(
        str(source), ctx=ctx, to=root, wait=True, processing_mode="vectors_only", build_index=False
    )
    child = root + "/independent"
    await fs.write_file(child + "/content.txt", "expire", ctx=ctx)
    await fs.write_file(
        child + "/.ttl.json",
        json.dumps({"expires_at": "2000-01-01T00:00:00.000Z", "ttl_generation": "old"}),
        ctx=ctx,
    )
    await fs.rm(child, recursive=True, strict=True, ctx=ctx)
    await fs.ttl_registry.remove_if_generation(ctx.account_id, child, "old")
    assert await fs.ttl_registry.get(ctx.account_id, child) is None
    result = await service.resources.refresh_resource(
        str(source), ctx=ctx, to=root, processing_mode="vectors_only", build_index=False
    )
    task = await get_task_tracker().wait(
        result["task_id"], account_id=ctx.account_id, user_id=ctx.user.user_id, timeout=25
    )
    assert task.status.value == "failed"
    assert "independently expiring" in task.error
    assert not await fs.exists(child, ctx=ctx)
