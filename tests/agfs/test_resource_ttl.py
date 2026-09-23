# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Resource TTL ownership and transfer with native AGFS leases and I/O."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.core import ttl
from openviking.service.ttl_cleanup import TTLCleanupService
from openviking.storage.ovpack.operations import export_ovpack, import_ovpack
from openviking.storage.resource_ttl import prepare_resource_ttl, resource_ttl_fields
from openviking_cli.exceptions import NotFoundError
from openviking_cli.utils.config.ttl_config import TTLConfig
from tests.storage.test_transfer_merge_binding import binding_fs as binding_fs
from tests.storage.test_transfer_merge_binding import root_ctx

ROOT = "viking://user/default/resources"
PAST = "2000-01-01T00:00:00.000Z"
FUTURE = "2999-01-01T00:00:00.000Z"


@pytest.fixture(autouse=True)
def disabled_policy(monkeypatch):
    monkeypatch.setattr(ttl, "get_openviking_config", lambda: SimpleNamespace(ttl=TTLConfig()))


async def set_expiry(fs, uri, *, is_dir=True, expiry=FUTURE):
    fields = {
        "expires_at": expiry,
        "received_at": "2020-01-01T00:00:00.000Z",
        "ttl_generation": "g1",
    }
    kind = ttl.OBJECT_TYPE_RESOURCE if is_dir else ttl.OBJECT_TYPE_RESOURCE_FILE
    await fs.write_file(ttl.ttl_metadata_uri(kind, uri), json.dumps(fields), ctx=root_ctx())
    return fields


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["cp", "mv"])
@pytest.mark.parametrize(
    "is_dir,inherited", [(True, False), (False, False), (True, True), (False, True)]
)
async def test_transfer_keeps_root_or_inherited_deadline(binding_fs, operation, is_dir, inherited):
    fs, ctx = binding_fs, root_ctx()
    source = ROOT + ("/owner/child" if inherited else "/source")
    target = ROOT + "/target"
    payload = source + "/part.txt" if is_dir else source
    await fs.write_file_bytes(payload, b"original bytes", ctx=ctx)
    if inherited:
        await set_expiry(fs, ROOT + "/owner", expiry="2998-01-01T00:00:00.000Z")
    if not inherited or is_dir:
        await set_expiry(fs, source, is_dir=is_dir)
    expected = await resource_ttl_fields(fs, source, ctx=ctx)
    await getattr(fs, operation)(
        source, target, ctx=ctx, **({"recursive": is_dir} if operation == "cp" else {})
    )
    assert (
        await fs.read_file_bytes(target + "/part.txt" if is_dir else target, ctx=ctx)
        == b"original bytes"
    )
    assert await resource_ttl_fields(fs, target, ctx=ctx) == expected
    record = await fs.ttl_registry.get(ctx.account_id, target)
    assert record.expires_at == expected["expires_at"]
    assert await fs.exists(source, ctx=ctx) == (operation == "cp")
    if operation == "mv" and not inherited:
        assert await fs.ttl_registry.get(ctx.account_id, source) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("is_dir", [True, False])
async def test_partial_cleanup_blocks_recreation_and_retry_removes_only_owner(
    binding_fs, monkeypatch, is_dir
):
    fs, ctx = binding_fs, root_ctx()
    uri, sibling = ROOT + "/expired", ROOT + "/keep"
    payload = uri + "/part.bin" if is_dir else uri
    await fs.write_file_bytes(payload, b"delete", ctx=ctx)
    await fs.write_file_bytes(sibling, b"keep", ctx=ctx)
    await set_expiry(fs, uri, is_dir=is_dir, expiry=PAST)
    record = await fs.ttl_registry.get(ctx.account_id, uri)
    cleanup = TTLCleanupService(
        service=SimpleNamespace(viking_fs=fs, fs=SimpleNamespace(rm=fs.rm)),
        service_loop=asyncio.get_running_loop(),
    )
    original = fs._confirm_fs_scope_cleared
    monkeypatch.setattr(
        fs,
        "_confirm_fs_scope_cleared",
        AsyncMock(side_effect=RuntimeError("confirmation unavailable")),
    )
    with pytest.raises(RuntimeError, match="confirmation unavailable"):
        await cleanup._cleanup_record(record)
    assert await fs.ttl_registry.get(ctx.account_id, uri) is not None
    with pytest.raises(NotFoundError):
        await prepare_resource_ttl(fs, uri, is_dir=is_dir, existing=False, ctx=ctx, lease_ref=None)
    monkeypatch.setattr(fs, "_confirm_fs_scope_cleared", original)
    assert (await cleanup._cleanup_record(record))["deleted"]
    assert await fs.ttl_registry.get(ctx.account_id, uri) is None
    assert await fs.read_file_bytes(sibling, ctx=ctx) == b"keep"
    metadata = ttl.ttl_metadata_uri("resource" if is_dir else "resource_file", uri)
    assert not await fs.exists(metadata, ctx=ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize("own_deadline", [False, True])
async def test_ovpack_subtree_inherits_deadline_and_cleanup_cannot_interleave(
    binding_fs, tmp_path, monkeypatch, own_deadline
):
    fs, ctx = binding_fs, root_ctx()
    owner, source = ROOT + "/owner", ROOT + "/owner/child"
    await fs.write_file_bytes(source + "/part.txt", b"original", ctx=ctx)
    await set_expiry(fs, owner, expiry="2998-01-01T00:00:00.000Z")
    if own_deadline:
        await set_expiry(fs, source)
    expected = await resource_ttl_fields(fs, source, ctx=ctx)
    archive = await export_ovpack(fs, source, str(tmp_path / "child.ovpack"), ctx)
    parent = ROOT + "/restored"
    await fs.mkdir(parent, ctx=ctx)
    target = parent + "/child"
    real_write = fs.write_file_bytes
    attempted_cleanup = []

    async def write(uri, data, **kwargs):
        await real_write(uri, data, **kwargs)
        if uri == target + "/.ttl.json":
            from openviking.storage.errors import LockAcquisitionError

            with pytest.raises(LockAcquisitionError):
                await fs._async_agfs.pathlock_acquire_tree(fs._uri_to_path(target, ctx=ctx))
            attempted_cleanup.append(True)

    monkeypatch.setattr(fs, "write_file_bytes", write)
    monkeypatch.setattr(
        "openviking.storage.ovpack.operations._enqueue_direct_vectorization", AsyncMock()
    )
    assert await import_ovpack(fs, archive, parent, ctx) == target
    assert attempted_cleanup
    assert await resource_ttl_fields(fs, target, ctx=ctx) == expected
    assert await fs.read_file_bytes(target + "/part.txt", ctx=ctx) == b"original"
    assert (await fs.ttl_registry.get(ctx.account_id, target)).generation == expected[
        "ttl_generation"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["copy", "vectors"])
async def test_failed_subtree_copy_keeps_only_published_ttl_records(
    binding_fs, monkeypatch, failure
):
    fs, ctx = binding_fs, root_ctx()
    source, target = ROOT + "/owner/child", ROOT + "/copied"
    await fs.write_file_bytes(source, b"body", ctx=ctx)
    await set_expiry(fs, ROOT + "/owner")
    if failure == "copy":
        original = fs._copy_agfs_entry

        async def fail_after_copy(*args, **kwargs):
            await original(*args, **kwargs)
            raise RuntimeError("copy interrupted")

        monkeypatch.setattr(fs, "_copy_agfs_entry", fail_after_copy)
    else:
        monkeypatch.setattr(
            fs,
            "_copy_vector_store_uris",
            AsyncMock(side_effect=RuntimeError("vectors unavailable")),
        )
    with pytest.raises(RuntimeError):
        await fs.cp(source, target, ctx=ctx)
    record = await fs.ttl_registry.get(ctx.account_id, target)
    if failure == "copy":
        assert record is not None
        assert await fs.read_file_bytes(target, ctx=ctx) == b"body"
    else:
        assert record is None
        assert await resource_ttl_fields(fs, target, ctx=ctx) == {}
        assert not await fs.exists(target, ctx=ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["live", "expired", "recreated", "updated"])
async def test_late_resource_embedding_validates_source_generation_and_content(
    binding_fs, monkeypatch, state
):
    from openviking.storage.collection_schemas import TextEmbeddingHandler
    from openviking.storage.queuefs.embedding_msg import EmbeddingMsg
    from openviking.utils.content_hash import content_md5

    fs, ctx = binding_fs, root_ctx()
    uri = ROOT + "/document"
    await fs.write_file_bytes(uri, b"updated" if state == "updated" else b"original", ctx=ctx)
    await set_expiry(fs, uri, is_dir=False, expiry=PAST if state == "expired" else FUTURE)
    monkeypatch.setattr("openviking.storage.viking_fs.get_viking_fs", lambda: fs)
    message = EmbeddingMsg(
        "original",
        {
            "uri": uri,
            "ttl_generation": "old" if state == "recreated" else "g1",
            "md5": content_md5(b"original"),
        },
    )
    write = AsyncMock(return_value="vector-id")
    result = await object.__new__(TextEmbeddingHandler)._write_ttl_vector_if_current(
        message, ctx, write
    )
    assert result == ("vector-id" if state == "live" else None)
    assert write.await_count == int(state == "live")
