# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Reindexing preserves event freshness in queued embeddings."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.service import reindex_executor as reindex_module
from openviking.service.reindex_executor import ReindexExecutor, _ReindexCounters
from openviking.storage.queuefs.embedding_msg import EmbeddingMsg
from openviking.storage.vikingdb_manager import VikingDBManager
from openviking.utils import embedding_utils
from openviking_cli.session.user_id import UserIdentifier


@pytest.mark.parametrize(
    ("mtime", "indexed_time", "expected"),
    [
        ("2026-01-02T00:00:00Z", "2025-01-01T00:00:00Z", "2026-01-02T00:00:00.000Z"),
        (None, "2025-01-01T00:00:00Z", "2025-01-01T00:00:00.000Z"),
        (None, "invalid", None),
    ],
)
async def test_memory_reindex_preserves_source_time_in_queue(
    monkeypatch, mtime, indexed_time, expected
):
    uri = "viking://user/alice/memories/events/old.md"
    existing = {"uri": uri, "abstract": "old event", "updated_at": indexed_time}
    fs = SimpleNamespace(
        exists=AsyncMock(return_value=True),
        stat=AsyncMock(return_value={"isDir": False, "modTime": mtime}),
        read_file=AsyncMock(return_value="old event" if mtime else ""),
    )

    async def project_record(**kwargs):
        return [{field: existing.get(field) for field in kwargs["output_fields"]}]

    # Exercise the real URI lookup so its projection cannot silently discard
    # the indexed time before reindexing uses it as the fallback.
    backend = SimpleNamespace(filter=AsyncMock(side_effect=project_record))
    manager = object.__new__(VikingDBManager)
    manager._get_backend_for_context = lambda _ctx: backend
    manager.enqueue_embedding_msg = AsyncMock(return_value=True)
    monkeypatch.setattr(
        reindex_module, "get_service", lambda: SimpleNamespace(vikingdb_manager=manager)
    )
    monkeypatch.setattr(reindex_module, "get_viking_fs", lambda: fs)
    monkeypatch.setattr(embedding_utils, "get_viking_fs", lambda: fs)
    # No directory sidecars are needed when the index already has the abstract.
    monkeypatch.setattr(ReindexExecutor, "_best_file_summary", AsyncMock(return_value=""))
    ctx = RequestContext(user=UserIdentifier("acc1", "alice"), role=Role.ROOT)
    monkeypatch.setattr(
        reindex_module,
        "get_openviking_config",
        lambda: SimpleNamespace(reindex=SimpleNamespace(file_vectorization_concurrency=1)),
    )
    counters = _ReindexCounters()

    await ReindexExecutor()._reindex_memory_vectors(uri=uri, counters=counters, ctx=ctx)

    assert counters.rebuilt_records == 1, counters.warnings
    assert "updated_at" in backend.filter.call_args.kwargs["output_fields"]
    msg = manager.enqueue_embedding_msg.call_args.args[0]
    # A queue round trip is also the retry payload; it must not introduce now.
    retried = EmbeddingMsg.from_json(msg.to_json())
    assert retried.context_data["updated_at"] == expected
