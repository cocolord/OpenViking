"""Resource TTL contracts: one import root, frozen lifetime and durable cleanup."""

import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.core import ttl
from openviking.server.identity import RequestContext, Role
from openviking.storage.resource_ttl import (
    prepare_resource_ttl,
    read_resource_fields,
    resource_ttl_visible,
    update_resource_expiry,
)
from openviking.storage.ttl_registry import TTLRegistry
from openviking.storage.viking_fs import VikingFS
from openviking_cli.exceptions import InvalidArgumentError, NotFoundError
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.ttl_config import ResourceTTL, TTLConfig
from tests.unit.service.test_ttl_cleanup import _make_service, _message, _record, _session_meta
from tests.unit.service.test_ttl_cleanup import tracker as tracker
from tests.unit.storage.test_ttl_registry import _MemoryAGFS

ROOT = "viking://user/u1/resources"
PAST = "2000-01-01T00:00:00.000Z"
FUTURE = "2999-01-01T00:00:00.000Z"


class MemoryAGFS(_MemoryAGFS):
    async def pathlock_acquire_exact(self, path, **kwargs):
        return await super().pathlock_acquire_exact(path)

    pathlock_acquire_tree = pathlock_acquire_exact

    async def ensure_parent_dirs(self, path, **kwargs):
        await super().ensure_parent_dirs(path)

    async def read(self, path, **kwargs):
        return await super().read(path)

    async def stat(self, path, **kwargs):
        self.stat_calls.append(path)
        if path in self.files:
            return {"isDir": False, "size": len(self.files[path])}
        if any(key.startswith(path.rstrip("/") + "/") for key in self.files):
            return {"isDir": True}
        raise FileNotFoundError(path)


@pytest.fixture
def fs_ctx(monkeypatch):
    monkeypatch.setattr(ttl, "get_openviking_config", lambda: SimpleNamespace(ttl=TTLConfig()))
    fs = VikingFS(agfs=SimpleNamespace())
    agfs = MemoryAGFS()
    fs._async_agfs = agfs
    fs.ttl_registry = TTLRegistry(agfs)
    fs._ensure_parent_dirs = AsyncMock()
    ctx = RequestContext(user=UserIdentifier("acct", "u1"), role=Role.ROOT)
    return fs, ctx


async def install(fs, ctx, uri, *, is_dir, expires_at=FUTURE):
    fields = {
        "received_at": "2020-01-01T00:00:00.000Z",
        "expires_at": expires_at,
        "ttl_generation": "g1",
    }
    kind = ttl.OBJECT_TYPE_RESOURCE if is_dir else ttl.OBJECT_TYPE_RESOURCE_FILE
    await fs.write_file(ttl.ttl_metadata_uri(kind, uri), json.dumps(fields), ctx=ctx)
    return fields


@pytest.mark.asyncio
async def test_retry_after_registry_only_write_keeps_pending_deadline(fs_ctx):
    fs, ctx = fs_ctx
    uri = ROOT + "/interrupted"
    expected = await install(fs, ctx, uri, is_dir=True)
    del fs._async_agfs.files[fs._uri_to_path(uri + "/.ttl.json", ctx=ctx)]
    fields = await prepare_resource_ttl(
        fs,
        uri,
        is_dir=True,
        existing=False,
        ctx=ctx,
        lease_ref=None,
        resource_ttl={"ttl_relative": 7},
    )
    assert fields["expires_at"] == expected["expires_at"]
    assert fields["ttl_generation"] == expected["ttl_generation"]
    assert await read_resource_fields(fs, "resource", uri, ctx=ctx) == fields


@pytest.mark.asyncio
@pytest.mark.parametrize("is_dir", [True, False])
@pytest.mark.parametrize(
    "root", [ROOT, "viking://resources", "viking://user/u1/peers/p1/resources"]
)
async def test_root_lifecycle_hides_bytes_and_registers_only_one_owner(fs_ctx, is_dir, root):
    fs, ctx = fs_ctx
    uri = root + "/doc"
    payload_uri = uri + "/sections/a.bin" if is_dir else uri
    await fs.write_file_bytes(payload_uri, b"\x00\xff original", ctx=ctx)
    await fs.write_file(ROOT + "/sibling", "keep", ctx=ctx)
    fields = await install(fs, ctx, uri, is_dir=is_dir)
    assert await fs.read_file_bytes(payload_uri, ctx=ctx) == b"\x00\xff original"
    assert (await fs.ttl_registry.get("acct", uri)).generation == "g1"
    if is_dir:
        assert await fs.ttl_registry.get("acct", payload_uri) is None
    fields["expires_at"] = PAST
    await install(fs, ctx, uri, is_dir=is_dir, expires_at=PAST)
    with pytest.raises(NotFoundError):
        await fs.read_file_bytes(payload_uri, ctx=ctx)
    assert await fs.read_file(ROOT + "/sibling", ctx=ctx) == "keep"
    # Removing metadata during a failed strict cleanup must not revive bytes.
    fs._async_agfs.files.pop(
        fs._uri_to_path(
            ttl.ttl_metadata_uri("resource" if is_dir else "resource_file", uri), ctx=ctx
        )
    )
    assert not await resource_ttl_visible(fs, payload_uri, ctx=ctx)


@pytest.mark.asyncio
async def test_new_root_freezes_but_reimport_and_children_keep_original(fs_ctx):
    fs, ctx = fs_ctx
    uri = ROOT + "/doc"
    first = await prepare_resource_ttl(
        fs,
        uri,
        is_dir=True,
        existing=False,
        ctx=ctx,
        lease_ref=None,
        resource_ttl={"ttl_relative": 7},
    )
    assert ttl.parse_iso_datetime(first["expires_at"]) - ttl.parse_iso_datetime(
        first["received_at"]
    ) == timedelta(days=7)
    assert (
        await prepare_resource_ttl(
            fs,
            uri,
            is_dir=True,
            existing=True,
            ctx=ctx,
            lease_ref=None,
            resource_ttl={"ttl_relative": 30},
        )
        == first
    )
    child = await prepare_resource_ttl(
        fs, uri + "/chapter.txt", is_dir=False, existing=False, ctx=ctx, lease_ref=None
    )
    assert child == first
    assert await fs.ttl_registry.get("acct", uri + "/chapter.txt") is None
    assert await read_resource_fields(fs, "resource_file", uri + "/chapter.txt", ctx=ctx) is None


@pytest.mark.asyncio
async def test_disabled_and_legacy_resources_do_not_gain_metadata(fs_ctx):
    fs, ctx = fs_ctx
    for existing, options in [(False, {}), (True, {"ttl_relative": 3})]:
        assert (
            await prepare_resource_ttl(
                fs,
                ROOT + "/legacy",
                is_dir=False,
                existing=existing,
                ctx=ctx,
                lease_ref=None,
                resource_ttl=options,
            )
            == {}
        )
    assert fs._async_agfs.files == {}


@pytest.mark.asyncio
async def test_expiry_edit_keeps_incarnation_and_updates_due_record(fs_ctx):
    fs, ctx = fs_ctx
    uri = ROOT + "/doc"
    await fs.write_file(uri + "/child", "text", ctx=ctx)
    before = await install(fs, ctx, uri, is_dir=True)
    fs.stat = AsyncMock(return_value={"isDir": True})
    after = await update_resource_expiry(fs, uri, "2998-01-01T00:00:00Z", ctx=ctx)
    assert after["ttl_generation"] == before["ttl_generation"]
    assert after["received_at"] == before["received_at"]
    assert (await fs.ttl_registry.get("acct", uri)).expires_at == after["expires_at"]
    with pytest.raises(InvalidArgumentError):
        await update_resource_expiry(fs, uri, "bad timestamp", ctx=ctx)


@pytest.mark.asyncio
async def test_resource_descendant_markers_cover_ancestors_not_siblings(fs_ctx):
    fs, ctx = fs_ctx
    await install(fs, ctx, ROOT + "/a/sub/doc", is_dir=True)
    assert await fs.ttl_registry.has_ttl_descendants("acct", ROOT + "/a/sub")
    assert await fs.ttl_registry.has_ttl_descendants("acct", ROOT + "/a")
    assert await fs.ttl_registry.has_ttl_descendants("acct", ROOT)
    assert not await fs.ttl_registry.has_ttl_descendants("acct", ROOT + "/b")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,recursive", [("resource", True), ("resource_file", False)])
async def test_cleanup_uses_common_strict_delete_and_persistent_retry(tracker, kind, recursive):
    record = _record(kind, object_uri=ROOT + "/doc")
    cleanup, fs, registry, queues = _make_service(
        record=record, live_content=_session_meta(), rm_error=RuntimeError("vector delete failed")
    )
    cleanup._service.fs = SimpleNamespace(rm=fs.rm)
    await cleanup._process(_message(record))
    registry.remove_if_generation.assert_not_awaited()
    registry.defer_retry.assert_awaited_once()
    assert fs.rm.await_args.kwargs["strict"] is True
    assert fs.rm.await_args.kwargs["recursive"] is recursive
    fs.rm.side_effect = None
    result = await cleanup._cleanup_record(record)
    assert result["deleted"]
    registry.remove_if_generation.assert_awaited_once()


@pytest.mark.parametrize(
    "values",
    [
        {"ttl_relative": 0},
        {"ttl_relative": True},
        {"ttl_relative": 1.5},
        {"ttl_absolute": 1.5},
        {"ttl_relative": 1, "ttl_absolute": 2000000000},
    ],
)
def test_public_ttl_rejects_ambiguous_or_non_integral_values(values):
    with pytest.raises(ValueError):
        ResourceTTL(**values)


def test_policy_nearest_and_per_import_override():
    config = TTLConfig(
        resources={"mode": "days", "ttl_days": 30},
        directories={ROOT + "/a": {"mode": "days", "ttl_days": 7}},
    )
    fields = ttl.freeze_ttl_fields(ROOT + "/a/doc", config=config)
    assert fields["ttl_days"] == 7
    exact = ttl.freeze_ttl_fields(
        ROOT + "/a/doc", config=config, resource_ttl={"ttl_absolute": 2000000000}
    )
    assert ttl.parse_iso_datetime(exact["expires_at"]).timestamp() == 2000000000
    assert config.resolve_uri(ROOT + "/ab/doc", "resources") == 30
