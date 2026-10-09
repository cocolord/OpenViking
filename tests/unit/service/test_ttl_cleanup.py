# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Cleanup rechecks the current deadline and reports failures without requeueing."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.pyagfs.exceptions import AGFSNetworkError, AGFSTimeoutError
from openviking.service import ttl_cleanup
from openviking.service.task_store import PersistentTaskStore
from openviking.service.task_tracker import TaskStatus, TaskTracker, set_task_tracker
from openviking.storage.queuefs.process_result import ProcessOutcome
from tests.storage.test_transfer_merge_binding import binding_fs as binding_fs
from tests.storage.test_transfer_merge_binding import root_ctx
from tests.unit.storage.ttl_test_storage import read_record


@pytest.fixture
async def cleanup_case(binding_fs):
    fs, ctx = binding_fs, root_ctx()
    owner = "viking://user/default/memories/events/2026/09/28"
    await fs.write_file(owner + "/body.md", "body", ctx=ctx)
    await fs.write_file(owner + "/.meta.json", '{"expires_at":"2000-01-01T00:00:00Z"}', ctx=ctx)
    record = await read_record(fs, ctx.account_id, owner)
    queue = AsyncMock()
    cleanup = ttl_cleanup.TTLCleanupService(
        service=SimpleNamespace(viking_fs=fs, _queue_manager=queue)
    )
    tracker = TaskTracker(PersistentTaskStore(fs._async_agfs))
    set_task_tracker(tracker)
    try:
        yield fs, ctx, cleanup, queue, tracker, record
    finally:
        set_task_tracker(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["extended", "disabled", "paused"])
async def test_pending_cleanup_obeys_current_deadline_and_pause(cleanup_case, monkeypatch, state):
    fs, ctx, cleanup, _, _, record = cleanup_case
    if state == "paused":
        monkeypatch.setattr(
            ttl_cleanup, "_cleanup_settings", lambda: SimpleNamespace(enabled=False)
        )
    else:
        expiry = "2999-01-01T00:00:00Z" if state == "extended" else None
        await fs.write_file(
            record.object_uri + "/.meta.json", json.dumps({"expires_at": expiry}), ctx=ctx
        )
    delete = AsyncMock()
    monkeypatch.setattr(fs, "rm", delete)
    result = await cleanup._process(ttl_cleanup._ttl_cleanup_message(record=record))
    assert result.outcome is ProcessOutcome.SUCCESS
    delete.assert_not_awaited()
    assert (
        await fs.read_file(record.object_uri + "/body.md", ctx=ctx, include_expired=True) == "body"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        AGFSNetworkError("endpoint not found"),
        AGFSTimeoutError("endpoint not found"),
        RuntimeError("delete failed"),
    ],
)
async def test_failed_attempt_is_reported_and_left_for_next_scan(
    cleanup_case, monkeypatch, failure
):
    fs, ctx, cleanup, queue, tracker, record = cleanup_case
    original_read = fs._async_agfs.read
    metadata = fs._uri_to_path(record.object_uri + "/.meta.json", ctx=ctx)

    async def read(path, **kwargs):
        if path == metadata:
            raise failure
        return await original_read(path, **kwargs)

    if isinstance(failure, (AGFSNetworkError, AGFSTimeoutError)):
        monkeypatch.setattr(fs._async_agfs, "read", read)
        delete = AsyncMock()
    else:
        delete = AsyncMock(side_effect=failure)
    monkeypatch.setattr(fs, "rm", delete)
    message = ttl_cleanup._ttl_cleanup_message(record=record)
    assert (await cleanup._process(message)).outcome is ProcessOutcome.FAILED
    task = await tracker.get(
        message["task_id"], account_id=message["account_id"], user_id=message["user_id"]
    )
    assert task.status is TaskStatus.FAILED and task.result is None
    assert not queue.mock_calls  # Retained metadata, not immediate requeueing, drives retries.
    assert await original_read(metadata)
    if isinstance(failure, (AGFSNetworkError, AGFSTimeoutError)):
        delete.assert_not_awaited()


async def _cleanup_once(cleanup, record):
    """Exercise the strict cleanup body under its required object lease."""
    ctx, lease = await cleanup._acquire_object_lock(record)
    try:
        return await cleanup._cleanup_record(record, ctx, lease)
    finally:
        await cleanup._service.viking_fs._async_agfs.pathlock_release(lease)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["acquire", "delete"])
async def test_contention_retry_rechecks_changed_deadline(cleanup_case, monkeypatch, stage):
    from openviking.storage.errors import LockAcquisitionError, ResourceBusyError

    fs, ctx, cleanup, queue, tracker, record = cleanup_case
    acquire = cleanup._acquire_object_lock
    release = AsyncMock(wraps=fs._async_agfs.pathlock_release)
    monkeypatch.setattr(fs._async_agfs, "pathlock_release", release)
    monkeypatch.setattr(ttl_cleanup.random, "uniform", lambda *args: 0)
    calls = 0

    async def contend_then_extend(target):
        nonlocal calls
        calls += 1
        if calls == 1 and stage == "acquire":
            raise LockAcquisitionError("busy")
        if calls == 2:
            if stage == "delete":
                assert release.await_count == 1
            await fs.write_file(
                record.object_uri + "/.meta.json",
                '{"expires_at":"2999-01-01T00:00:00Z"}',
                ctx=ctx,
            )
        return await acquire(target)

    monkeypatch.setattr(cleanup, "_acquire_object_lock", contend_then_extend)
    delete = AsyncMock(side_effect=ResourceBusyError("busy", uri=record.object_uri))
    monkeypatch.setattr(fs, "rm", delete)
    result = await cleanup._process(ttl_cleanup._ttl_cleanup_message(record=record))
    assert result.outcome is ProcessOutcome.SUCCESS
    assert result.value == {"deleted": False, "skipped": "live_or_unmanaged"}
    assert calls == 2
    assert delete.await_count == int(stage == "delete")
    assert not queue.mock_calls


@pytest.mark.asyncio
async def test_persistent_contention_is_bounded_and_preserves_deadline(cleanup_case, monkeypatch):
    from openviking.storage.errors import ResourceBusyError

    fs, ctx, cleanup, queue, tracker, record = cleanup_case
    monkeypatch.setattr(ttl_cleanup.random, "uniform", lambda *args: 0)
    delete = AsyncMock(side_effect=ResourceBusyError("busy", uri=record.object_uri))
    monkeypatch.setattr(fs, "rm", delete)
    release = AsyncMock(wraps=fs._async_agfs.pathlock_release)
    monkeypatch.setattr(fs._async_agfs, "pathlock_release", release)
    message = ttl_cleanup._ttl_cleanup_message(record=record)
    result = await cleanup._process(message)
    assert result.outcome is ProcessOutcome.SUCCESS
    assert delete.await_count == release.await_count == 3
    task = await tracker.get(
        message["task_id"], account_id=message["account_id"], user_id=message["user_id"]
    )
    assert task.result == {"deleted": False, "skipped": "busy"}
    assert not queue.mock_calls
    assert b"2000-01-01" in await fs._async_agfs.read(
        fs._uri_to_path(record.object_uri + "/.meta.json", ctx=ctx)
    )
