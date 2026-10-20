# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import asyncio

import pytest

from openviking.service.task_work_index import bind_task_context, get_task_context
from openviking.session.memory.utils.streaming_batcher import (
    StreamingBatcher,
    StreamingBatcherConfig,
)


@pytest.mark.asyncio
async def test_timer_flush_runs_without_submitter_context():
    seen_task_ids: list[str | None] = []

    async def process_batch(items: list[str], _reason: str) -> str:
        context = get_task_context()
        seen_task_ids.append(context.task_id if context is not None else None)
        return ",".join(items)

    batcher = StreamingBatcher(
        name="context-test",
        process_batch=process_batch,
        config=StreamingBatcherConfig(
            max_items_per_batch=10,
            max_wait_seconds=0.01,
            timer_check_interval_seconds=0.001,
        ),
    )
    with bind_task_context("first-task", "account", "user"):
        assert await asyncio.wait_for(batcher.submit("first"), timeout=1) == "first"

    assert await asyncio.wait_for(batcher.submit("second"), timeout=1) == "second"
    await batcher.close()

    assert seen_task_ids == [None, None]


@pytest.mark.asyncio
async def test_count_flush_runs_without_last_submitter_context():
    seen_task_ids: list[str | None] = []

    async def process_batch(items: list[str], _reason: str) -> str:
        context = get_task_context()
        seen_task_ids.append(context.task_id if context is not None else None)
        return ",".join(items)

    batcher = StreamingBatcher(
        name="context-test",
        process_batch=process_batch,
        config=StreamingBatcherConfig(
            max_items_per_batch=2,
            max_wait_seconds=10,
            timer_check_interval_seconds=1,
        ),
    )
    with bind_task_context("first-task", "account", "user"):
        first = asyncio.create_task(batcher.submit("first"))
    await asyncio.sleep(0)
    assert await batcher.get_buffered_item_count() == 1

    with bind_task_context("last-task", "account", "user"):
        second = asyncio.create_task(batcher.submit("second"))

    assert await asyncio.wait_for(first, timeout=1) == "first,second"
    assert await asyncio.wait_for(second, timeout=1) == "first,second"
    await batcher.close()

    assert seen_task_ids == [None]


@pytest.mark.asyncio
async def test_close_waits_for_in_flight_timer_flush():
    started = asyncio.Event()
    finish = asyncio.Event()

    async def process_batch(items: list[str], _reason: str) -> str:
        started.set()
        await finish.wait()
        return ",".join(items)

    batcher = StreamingBatcher(
        name="test",
        process_batch=process_batch,
        config=StreamingBatcherConfig(
            max_items_per_batch=10,
            max_wait_seconds=0.01,
            timer_check_interval_seconds=0.001,
        ),
    )
    submit = asyncio.create_task(batcher.submit("item"))
    await asyncio.wait_for(started.wait(), timeout=1)

    close = asyncio.create_task(batcher.close())
    await asyncio.sleep(0)
    finish.set()

    assert await asyncio.wait_for(submit, timeout=1) == "item"
    assert await asyncio.wait_for(close, timeout=1) is None


@pytest.mark.asyncio
async def test_close_drains_failing_timer_flush_without_unhandled_task():
    started = asyncio.Event()
    finish = asyncio.Event()
    unhandled: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()

    async def process_batch(_items: list[str], _reason: str) -> str:
        started.set()
        await finish.wait()
        raise RuntimeError("boom")

    batcher = StreamingBatcher(
        name="failing-test",
        process_batch=process_batch,
        config=StreamingBatcherConfig(
            max_items_per_batch=10,
            max_wait_seconds=0.01,
            timer_check_interval_seconds=0.001,
        ),
    )
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        submit = asyncio.create_task(batcher.submit("item"))
        await asyncio.wait_for(started.wait(), timeout=1)

        close = asyncio.create_task(batcher.close())
        await asyncio.sleep(0)
        finish.set()

        with pytest.raises(RuntimeError, match="boom"):
            await asyncio.wait_for(submit, timeout=1)
        assert await asyncio.wait_for(close, timeout=1) is None
        await asyncio.sleep(0)

        assert submit.done()
        assert close.done()
        assert unhandled == []
    finally:
        loop.set_exception_handler(previous_handler)
