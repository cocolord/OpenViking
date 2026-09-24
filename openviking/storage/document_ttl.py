# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Explicit expiry edits for live event and resource files."""

from openviking.core.ttl import (
    OBJECT_TYPE_EVENT,
    OBJECT_TYPE_RESOURCE_FILE,
    TTL_FIELD_NAMES,
    hidden_by_ttl,
    ttl_object_for_uri,
    ttl_scope_for_uri,
)
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.storage.acl import AclAction
from openviking.storage.resource_ttl import (
    read_resource_fields,
    resource_ttl_fields,
    resource_ttl_targets,
    write_resource_fields,
)
from openviking.utils.time_utils import format_iso8601, parse_iso_datetime
from openviking_cli.exceptions import ConflictError, InvalidArgumentError, NotFoundError


async def _document_target(fs, uri, *, ctx):
    stat = await fs.stat(uri, ctx=ctx)
    if ttl_scope_for_uri(uri) == "resources":
        if stat.get("isDir"):
            raise InvalidArgumentError(
                "resource directories define defaults via resources/config; "
                "resources/ttl requires a file"
            )
        for kind, owner in resource_ttl_targets(uri):
            fields = await read_resource_fields(fs, kind, owner, ctx=ctx)
            if fields is not None:
                return kind, owner, fields
        return OBJECT_TYPE_RESOURCE_FILE, uri, {}
    if ttl_object_for_uri(uri, is_dir=bool(stat.get("isDir"))) == (OBJECT_TYPE_EVENT, uri):
        memory = MemoryFileUtils.read(await fs.read_file(uri, ctx=ctx), uri=uri)
        return (
            OBJECT_TYPE_EVENT,
            uri,
            {key: value for key, value in memory.extra_fields.items() if key in TTL_FIELD_NAMES},
        )
    raise InvalidArgumentError("uri must identify an event file or resource document")


async def get_document_ttl(fs, uri: str, *, ctx) -> dict:
    kind, owner, fields = await _document_target(fs, uri, ctx=ctx)
    # Sidecars can carry private lifecycle bookkeeping (for example a Watch
    # tombstone fingerprint). Keep the public API limited to TTL fields.
    result = {
        "uri": uri,
        **{key: value for key, value in fields.items() if key in TTL_FIELD_NAMES},
    }
    if owner != uri:
        result["owner_uri"] = owner
    if kind != OBJECT_TYPE_EVENT:
        effective = await resource_ttl_fields(fs, uri, ctx=ctx)
        if effective.get("expires_at") != fields.get("expires_at"):
            result["owner_expires_at"] = fields.get("expires_at")
            result["expires_at"] = effective.get("expires_at")
    return result


async def update_document_expiry(fs, uri: str, expires_at: str, *, ctx) -> dict:
    """Reuse source locks and registry-first writes; never resurrect expired objects."""
    try:
        expiry = format_iso8601(parse_iso_datetime(expires_at))
    except (ValueError, TypeError) as exc:
        raise InvalidArgumentError("expires_at must be an ISO 8601 timestamp") from exc
    if hidden_by_ttl(expiry):
        raise InvalidArgumentError("expires_at must be in the future")
    kind, owner, original = await _document_target(fs, uri, ctx=ctx)
    await fs._ensure_access(owner, ctx, action=AclAction.WRITE)
    if not original.get("expires_at") or not original.get("ttl_generation"):
        raise InvalidArgumentError("document has no frozen TTL to update")
    lease = await fs._async_agfs.pathlock_acquire_exact(fs._uri_to_path(owner, ctx=ctx))
    try:
        live_kind, live_owner, fields = await _document_target(fs, uri, ctx=ctx)
        if (live_kind, live_owner, fields.get("ttl_generation")) != (
            kind,
            owner,
            original["ttl_generation"],
        ):
            raise ConflictError("document changed while updating its expiry; reload and retry")
        if hidden_by_ttl(fields["expires_at"]):
            raise NotFoundError(uri, "document")
        if kind == OBJECT_TYPE_EVENT:
            memory = MemoryFileUtils.read(await fs.read_file(uri, ctx=ctx), uri=uri)
            memory.extra_fields["expires_at"] = expiry
            await fs.write_file(uri, MemoryFileUtils.write(memory), ctx=ctx, lease_ref=lease)
        else:
            fields["expires_at"] = expiry
            await write_resource_fields(fs, kind, owner, fields, ctx=ctx, lease_ref=lease)
    finally:
        await fs._async_agfs.pathlock_release(lease)
    return await get_document_ttl(fs, uri, ctx=ctx)
