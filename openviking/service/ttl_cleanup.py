# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Durable physical cleanup for TTL-expired events and sessions.

Visibility is enforced synchronously by the read barriers.  This service only
does the slower physical half: it walks the small TTL registry, schedules due
records on QueueFS, and removes one exact object incarnation under its path
lock.  A task completes only after filesystem data, vector records, and its
registry record are all gone.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import NAMESPACE_URL, uuid5

from openviking.core.ttl import (
    OBJECT_TYPE_EVENT,
    OBJECT_TYPE_RESOURCE,
    OBJECT_TYPE_RESOURCE_FILE,
    OBJECT_TYPE_SESSION,
    hidden_by_ttl,
    ttl_metadata_uri,
)
from openviking.server.error_mapping import is_storage_not_found
from openviking.server.identity import RequestContext, Role
from openviking.service.periodic_task import PeriodicTask
from openviking.service.task_store import SYSTEM_TASK_ACCOUNT_ID, SYSTEM_TASK_USER_ID
from openviking.service.task_tracker import TaskStatus, get_task_tracker
from openviking.service.task_tracker_concurrency import OwnerLoopDispatcher, run_to_completion
from openviking.service.task_work_index import detach_task_context
from openviking.session.memory.utils.messages import parse_memory_file_with_fields
from openviking.session.ttl_fence import reconcile_session_ttl
from openviking.storage.queuefs.named_queue import DequeueHandlerBase
from openviking.storage.queuefs.process_result import ProcessResult
from openviking.storage.queuefs.semantic_msg import SemanticMsg, build_semantic_coalesce_key
from openviking.storage.ttl_registry import TTLRecord
from openviking.utils.time_utils import format_iso8601
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils import VikingURI
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


def _cleanup_task_id(record: TTLRecord, *, attempt: int = 0) -> str:
    """Return a stable task id for one scheduled expiry revision.

    Session renewal keeps the incarnation generation but advances expires_at.
    Including both prevents a completed pre-renewal no-op from suppressing the
    next legitimate cleanup while still deduplicating duplicate scan passes.
    """
    key = (
        f"openviking:ttl:{record.account_id}:{record.object_uri}:"
        f"{record.generation}:{record.expires_at}:{max(0, attempt)}"
    )
    return str(uuid5(NAMESPACE_URL, key))


def _ttl_cleanup_message(
    *,
    record: TTLRecord,
    task_id: Optional[str] = None,
    retry_count: int = 0,
) -> dict[str, Any]:
    return {
        "task_id": task_id or _cleanup_task_id(record, attempt=retry_count),
        "account_id": SYSTEM_TASK_ACCOUNT_ID,
        "user_id": SYSTEM_TASK_USER_ID,
        "target": asdict(record),
        "retry_count": max(0, int(retry_count)),
    }


class TTLCleanupService:
    """Own the TTL cleanup consumer and registry scheduler."""

    def __init__(
        self,
        *,
        service: Any,
        service_loop: asyncio.AbstractEventLoop,
        check_interval: Optional[float] = None,
    ) -> None:
        self._service = service
        self._service_loop = service_loop
        self._scheduler = TTLCleanupScheduler(service, check_interval=check_interval)

    async def initialize(self) -> None:
        queue_manager = self._service._queue_manager
        queue = queue_manager.get_queue(queue_manager.TTL_CLEANUP)
        queue.set_dequeue_handler(_TTLCleanupProcessor(self, self._service_loop))
        await self._scheduler.start()

    async def close(self) -> None:
        await self._scheduler.stop()

    async def _process(self, message: dict[str, Any]) -> ProcessResult:
        """Settle or durably replace one cleanup delivery."""
        task_id = message["task_id"]
        owner = {"account_id": message["account_id"], "user_id": message["user_id"]}
        record = TTLRecord.from_dict(message["target"])

        tracker = get_task_tracker()
        task = await tracker.create(
            "ttl_cleanup",
            resource_id=record.object_uri,
            task_id=task_id,
            **owner,
        )
        # Only a completed cleanup proves that physical data and the registry
        # projection are both gone. A persisted FAILED/CANCELLED task from an
        # older process must not make a recovered QueueFS delivery ACK without
        # retrying the strict delete.
        if task.status is TaskStatus.COMPLETED:
            return ProcessResult.success()
        if task.status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
            # TaskTracker terminal states are immutable.  A delivery restored
            # from an older implementation therefore needs a fresh task id;
            # otherwise physical cleanup may succeed while the persisted task
            # remains permanently failed/cancelled.  The deterministic attempt
            # id keeps duplicate recovery deliveries idempotent.
            retry_count = int(message.get("retry_count", 0)) + 1
            replacement = _ttl_cleanup_message(
                record=record,
                retry_count=retry_count,
            )
            queue_manager = self._service._queue_manager
            await queue_manager.enqueue(queue_manager.TTL_CLEANUP, replacement)
            return ProcessResult.requeued()

        await tracker.start(task_id, stage="strict_cleanup", **owner)
        try:
            result = await run_to_completion(lambda: self._cleanup_record(record))
        except Exception as exc:
            # QueueFS ACKs every returned result, including FAILED. Persist a
            # delayed retry first and return REQUEUED so partial deletion or
            # eventual vector consistency cannot strand an uncleared object.
            error = f"TTL cleanup retry: {exc}"
            await tracker.update_stage(
                task_id,
                "retrying",
                meta={
                    "last_error": error,
                    "retry_count": int(message.get("retry_count", 0)) + 1,
                },
                **owner,
            )
            retry_count = int(message.get("retry_count", 0)) + 1
            delay = min(3600.0, 30.0 * (2 ** min(retry_count - 1, 7)))
            next_retry_at = format_iso8601(
                datetime.now(timezone.utc)
                + timedelta(seconds=min(3600.0, delay * random.uniform(1.0, 1.2)))
            )
            # Persist before ACK, keeping delayed retries out of the immediate queue.
            deferred = await self._service.viking_fs.ttl_registry.defer_retry(
                record,
                retry_count=retry_count,
                task_id=task_id,
                next_retry_at=next_retry_at,
            )
            if not deferred:
                await tracker.complete(
                    task_id, {"deleted": False, "skipped": "superseded"}, **owner
                )
                return ProcessResult.success()
            logger.warning(
                "TTL cleanup requeued for %s generation=%s: %s",
                record.object_uri,
                record.generation,
                exc,
            )
            return ProcessResult.requeued()

        await tracker.complete(task_id, result, **owner)
        return ProcessResult.success()

    async def _cleanup_record(self, scheduled: TTLRecord) -> dict[str, Any]:
        """Strictly delete one generation while holding its object lock."""
        viking_fs = self._service.viking_fs
        registry = viking_fs.ttl_registry
        ctx = RequestContext(
            user=UserIdentifier(scheduled.account_id, scheduled.user_id or SYSTEM_TASK_USER_ID),
            role=Role.ROOT,
        )
        object_path = viking_fs._uri_to_path(scheduled.object_uri, ctx=ctx)
        if scheduled.object_type == OBJECT_TYPE_SESSION:
            lease = await viking_fs._async_agfs.pathlock_acquire_tree(object_path)
        else:
            parent_uri = VikingURI(scheduled.object_uri).parent.uri
            lease = await viking_fs._async_agfs.pathlock_acquire_batch(
                [
                    {
                        "path": object_path,
                        "kind": "tree"
                        if scheduled.object_type == OBJECT_TYPE_RESOURCE
                        else "exact",
                    },
                    *(
                        [
                            {
                                "path": viking_fs._uri_to_path(
                                    ttl_metadata_uri(scheduled.object_type, scheduled.object_uri),
                                    ctx=ctx,
                                ),
                                "kind": "exact",
                            }
                        ]
                        if scheduled.object_type == OBJECT_TYPE_RESOURCE_FILE
                        else []
                    ),
                    {
                        "path": viking_fs._uri_to_path(f"{parent_uri}/.abstract.md", ctx=ctx),
                        "kind": "exact",
                    },
                    {
                        "path": viking_fs._uri_to_path(f"{parent_uri}/.overview.md", ctx=ctx),
                        "kind": "exact",
                    },
                ]
            )
        try:
            registered = await registry.get(scheduled.account_id, scheduled.object_uri)
            if registered is None or registered.generation != scheduled.generation:
                return {"deleted": False, "skipped": "stale_registry_generation"}

            if scheduled.object_type == OBJECT_TYPE_SESSION:
                await reconcile_session_ttl(
                    viking_fs,
                    ctx,
                    session_uri=scheduled.object_uri,
                    generation=scheduled.generation,
                    lease_ref=lease,
                )
            live = await self._read_live_record(scheduled, ctx)
            if live is not None and live.generation != scheduled.generation:
                # An import/restore may have replaced the source without going
                # through the normal registry-first writer.  Repair the
                # projection when the replacement has its own complete TTL
                # snapshot; otherwise discard only the stale old projection.
                if live.generation and live.expires_at:
                    await registry.upsert(live)
                else:
                    await registry.remove_if_generation(
                        scheduled.account_id, scheduled.object_uri, scheduled.generation
                    )
                return {"deleted": False, "skipped": "stale_object_generation"}
            if live is not None and not hidden_by_ttl(live.expires_at):
                if live.expires_at != registered.expires_at:
                    await registry.upsert(live)
                return {"deleted": False, "skipped": "renewed"}

            # Missing source still requires strict vector cleanup.  Passing the
            # already-held lease makes the live re-check and the whole delete
            # one critical section; writers cannot renew or recreate between.
            remove = (
                self._service.fs.rm
                if scheduled.object_type in {OBJECT_TYPE_RESOURCE, OBJECT_TYPE_RESOURCE_FILE}
                else viking_fs.rm
            )
            await remove(
                scheduled.object_uri,
                recursive=scheduled.object_type in {OBJECT_TYPE_SESSION, OBJECT_TYPE_RESOURCE},
                ctx=ctx,
                lease_ref=lease,
                strict=True,
            )
            if scheduled.object_type != OBJECT_TYPE_SESSION:
                await self._invalidate_event_parent(
                    event_uri=scheduled.object_uri,
                    ctx=ctx,
                    lease=lease,
                )
            removed = await registry.remove_if_generation(
                scheduled.account_id, scheduled.object_uri, scheduled.generation
            )
            if not removed:
                raise RuntimeError(
                    f"TTL registry record changed before cleanup completion: {scheduled.object_uri}"
                )
            # The only process-local VikingFS cache stores count-based engine
            # selection hints.  Clear it after physical deletion so no stale
            # scope count survives cleanup.
            viking_fs._count_cache.clear()
            return {"deleted": True, "source_missing": live is None}
        finally:
            await viking_fs._async_agfs.pathlock_release(lease)

    async def _invalidate_event_parent(
        self,
        *,
        event_uri: str,
        ctx: RequestContext,
        lease: Any,
    ) -> None:
        """Remove shared event summaries/vectors and queue a fresh rebuild.

        The caller holds one batch lease over the event and both sidecars. A
        newer coalesced semantic message invalidates any older in-flight
        summary before the files and exact parent L0/L1 vectors are removed.
        Rebuild is asynchronous; invalidation itself is part of strict cleanup.
        """
        from openviking.core.namespace import context_type_for_uri

        context_type = context_type_for_uri(event_uri)
        parent_uri = VikingURI(event_uri).parent.uri
        queue_manager = self._service._queue_manager
        semantic_queue = queue_manager.get_queue(queue_manager.SEMANTIC, allow_create=True)
        semantic_msg = SemanticMsg(
            uri=parent_uri,
            context_type=context_type,
            recursive=False,
            account_id=ctx.account_id,
            user_id=ctx.user.user_id,
            group_ids=ctx.group_ids,
            role=str(ctx.role),
            changes={"deleted": [event_uri]},
            generation_trigger="ttl_cleanup",
            coalesce_key=build_semantic_coalesce_key(
                context_type=context_type,
                uri=parent_uri,
                account_id=ctx.account_id,
                user_id=ctx.user.user_id,
            ),
        )
        sidecars = [f"{parent_uri}/.abstract.md", f"{parent_uri}/.overview.md"]
        for sidecar_uri in sidecars:
            await self._service.viking_fs.rm(
                sidecar_uri,
                recursive=False,
                ctx=ctx,
                lease_ref=lease,
                strict=True,
            )
        await self._service.viking_fs._delete_from_vector_store([parent_uri], ctx=ctx)
        await self._service.viking_fs._confirm_vector_uris_cleared([parent_uri], ctx=ctx)
        # Rebuilding the still-live siblings is maintenance after invalidation,
        # not part of proving this object's physical deletion.  Keep it outside
        # the cleanup task so a later LLM failure cannot turn a completed strict
        # cleanup into FAILED or delay its completion.
        with detach_task_context():
            await semantic_queue.enqueue(semantic_msg)

    async def _read_live_record(
        self, scheduled: TTLRecord, ctx: RequestContext
    ) -> Optional[TTLRecord]:
        viking_fs = self._service.viking_fs
        read_uri = ttl_metadata_uri(scheduled.object_type, scheduled.object_uri)
        try:
            raw = await viking_fs.read_file(read_uri, ctx=ctx, include_expired=True)
        except Exception as exc:
            if is_storage_not_found(exc):
                return None
            raise
        if scheduled.object_type != OBJECT_TYPE_EVENT:
            fields = json.loads(raw)
        else:
            fields = parse_memory_file_with_fields(raw)
        if not isinstance(fields, dict):
            raise ValueError(f"Invalid TTL metadata for {scheduled.object_uri}")
        return TTLRecord(
            object_uri=scheduled.object_uri,
            object_type=scheduled.object_type,
            account_id=scheduled.account_id,
            user_id=scheduled.user_id,
            expires_at=str(fields.get("expires_at") or ""),
            generation=str(fields.get("ttl_generation") or ""),
        )


class TTLCleanupScheduler(PeriodicTask):
    """Periodically enqueue due records from the persistent TTL registry."""

    DEFAULT_CHECK_INTERVAL = 30.0

    def __init__(
        self,
        service: Any,
        *,
        check_interval: Optional[float] = None,
        sleep: Any = asyncio.sleep,
    ) -> None:
        self._service = service
        self._check_interval = (
            self.DEFAULT_CHECK_INTERVAL if check_interval is None else float(check_interval)
        )
        super().__init__(interval=self._check_interval, sleep=sleep)

    async def _scan_once(self) -> None:
        queue_manager = self._service._queue_manager
        queue = queue_manager.get_queue(queue_manager.TTL_CLEANUP)
        scheduled = 0
        async for record, payload in self._service.viking_fs.ttl_registry.claim_due(
            now=datetime.now(timezone.utc),
            limit=100,
            max_bytes=1_048_576,
            time_budget=5.0,
        ):
            await queue.enqueue(
                _ttl_cleanup_message(
                    record=record,
                    task_id=payload.get("task_id"),
                    retry_count=int(payload.get("retry_count", 0)),
                )
            )
            scheduled += 1
        if scheduled:
            logger.info("TTLCleanupScheduler scheduled=%d", scheduled)


class _TTLCleanupProcessor(DequeueHandlerBase):
    """Single-consumer QueueFS bridge into the service owner loop."""

    def __init__(
        self, cleanup_service: TTLCleanupService, service_loop: asyncio.AbstractEventLoop
    ) -> None:
        self._cleanup_service = cleanup_service
        self._dispatcher = OwnerLoopDispatcher(service_loop)

    @staticmethod
    def _parse_message(data: dict[str, Any]) -> dict[str, Any]:
        payload = data.get("data", data)
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            raise ValueError("Invalid TTL cleanup message")
        target = payload.get("target")
        if not isinstance(target, dict):
            raise ValueError("Invalid TTL cleanup target")
        required_owner = (
            payload.get("task_id") and payload.get("account_id") and payload.get("user_id")
        )
        if not required_owner:
            raise ValueError("Invalid TTL cleanup owner")
        object_type = str(target.get("object_type") or "")
        object_uri = str(target.get("object_uri") or "")
        generation = str(target.get("generation") or "")
        if object_type not in (
            OBJECT_TYPE_EVENT,
            OBJECT_TYPE_SESSION,
            OBJECT_TYPE_RESOURCE,
            OBJECT_TYPE_RESOURCE_FILE,
        ):
            raise ValueError("Invalid TTL cleanup object type")
        if not object_uri or not generation:
            raise ValueError("Invalid TTL cleanup object fence")
        return {
            "task_id": str(payload["task_id"]),
            "account_id": str(payload["account_id"]),
            "user_id": str(payload["user_id"]),
            "retry_count": max(0, int(payload.get("retry_count", 0))),
            "target": {
                "object_type": object_type,
                "object_uri": object_uri,
                "account_id": str(target.get("account_id") or ""),
                "user_id": str(target.get("user_id") or ""),
                "expires_at": str(target.get("expires_at") or ""),
                "generation": generation,
            },
        }

    async def on_dequeue(self, data: Optional[dict[str, Any]]) -> ProcessResult:
        if not data:
            return ProcessResult.success()
        try:
            message = self._parse_message(data)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return ProcessResult.failed(str(exc))
        return await self._dispatcher.run(lambda: self._cleanup_service._process(message))

    async def on_cancelled(self, data: Optional[dict[str, Any]]) -> ProcessResult:
        """Recover legacy terminal cleanup tasks instead of dropping work."""
        return await self.on_dequeue(data)


async def setup_ttl_cleanup(*, service: Any) -> Optional[TTLCleanupService]:
    if service.viking_fs is None or service._queue_manager is None:
        return None
    cleanup_service = TTLCleanupService(
        service=service,
        service_loop=asyncio.get_running_loop(),
    )
    await cleanup_service.initialize()
    service._ttl_cleanup_service = cleanup_service
    return cleanup_service
