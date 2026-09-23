# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Resource source metadata adapter for the common TTL lifecycle.

A parsed document owns its root subtree; a flat file owns only its own sidecar.
Source bytes and generated summaries never serve as lifecycle metadata.
"""

from __future__ import annotations

import json

from openviking.core.namespace import classify_uri
from openviking.core.ttl import (
    OBJECT_TYPE_RESOURCE,
    OBJECT_TYPE_RESOURCE_FILE,
    freeze_ttl_fields,
    hidden_by_ttl,
    ttl_enabled,
    ttl_metadata_uri,
    ttl_scope_for_uri,
)
from openviking.pyagfs.exceptions import AGFSNotADirectoryError
from openviking.server.error_mapping import is_storage_not_found
from openviking.utils.time_utils import format_iso8601, parse_iso_datetime
from openviking_cli.exceptions import ConflictError, InvalidArgumentError, NotFoundError


def resource_ttl_targets(uri: str):
    """Yield the exact file and containing document roots, nearest first."""
    if ttl_scope_for_uri(uri) != "resources":
        return
    shape = classify_uri(uri)
    parts = shape.parts
    root_depth = (shape.content_index or 0) + 1
    if len(parts) <= root_depth:
        return
    yield OBJECT_TYPE_RESOURCE_FILE, uri.rstrip("/")
    for depth in range(len(parts), root_depth, -1):
        yield OBJECT_TYPE_RESOURCE, "viking://" + "/".join(parts[:depth])


async def read_resource_fields(fs, object_type: str, uri: str, *, ctx):
    path = fs._uri_to_path(ttl_metadata_uri(object_type, uri), ctx=ctx)
    try:
        # Never cache absence: imports/restores can publish metadata on another worker.
        await fs._async_agfs.stat(path, bypass_cache=True)
        raw = fs._handle_agfs_read(await fs._async_agfs.read(path))
    except (NotADirectoryError, AGFSNotADirectoryError):
        # The candidate owner may be a flat file, which has no child sidecar.
        return None
    except Exception as exc:
        if is_storage_not_found(exc):
            return None
        raise
    fields = json.loads(raw)
    if not isinstance(fields, dict) or not fields.get("ttl_generation"):
        raise ValueError(f"Invalid resource TTL metadata for {uri}")
    fields["expires_at"] = format_iso8601(parse_iso_datetime(fields["expires_at"]))
    return fields


async def resource_ttl_fields(fs, uri: str, *, ctx) -> dict:
    """Return the nearest object's incarnation; parents still constrain visibility."""
    if not ttl_enabled() and not await fs.ttl_registry.account_may_have_records(ctx.account_id):
        return {}
    result = {}
    for object_type, owner in resource_ttl_targets(uri):
        fields = await read_resource_fields(fs, object_type, owner, ctx=ctx)
        if fields is None:
            continue
        if not result:
            result = dict(fields)
        expiry = fields.get("expires_at")
        if expiry and (not result.get("expires_at") or expiry < result["expires_at"]):
            result["expires_at"] = expiry
    return result


async def resource_ttl_visible(fs, uri: str, *, ctx, require_source=False) -> bool:
    for object_type, owner in resource_ttl_targets(uri):
        fields = await read_resource_fields(fs, object_type, owner, ctx=ctx)
        if fields is not None:
            if hidden_by_ttl(fields.get("expires_at")):
                return False
        else:
            # Partial strict deletion may remove metadata before all bytes/index rows.
            record = await fs.ttl_registry.get(ctx.account_id, owner)
            if record is not None and hidden_by_ttl(record.expires_at):
                return False
    if require_source:
        try:
            await fs._async_agfs.stat(fs._uri_to_path(uri, ctx=ctx), bypass_cache=True)
        except Exception as exc:
            if is_storage_not_found(exc):
                return False
            raise
    return True


async def prepare_resource_ttl(
    fs, uri: str, *, is_dir: bool, existing: bool, ctx, lease_ref, resource_ttl=None
) -> dict:
    """Publish metadata/registration before content under the import's source lease.

    Re-importing preserves the old snapshot, even when it had no TTL. Only a new
    resource freezes current policy. A failed content write leaves retryable
    metadata, so retrying cannot accidentally extend the deadline.
    """
    if ttl_scope_for_uri(uri) != "resources":
        return {}
    object_type = OBJECT_TYPE_RESOURCE if is_dir else OBJECT_TYPE_RESOURCE_FILE
    # Check the object itself as well as its ancestors: partial cleanup can
    # leave a tombstone after removing metadata, while vector deletion retries.
    if not await resource_ttl_visible(fs, uri, ctx=ctx):
        raise NotFoundError(uri, "resource")
    fields = await read_resource_fields(fs, object_type, uri, ctx=ctx)
    if fields is not None:
        return fields
    pending = await fs.ttl_registry.get(ctx.account_id, uri)
    if pending is not None:
        if pending.object_type != object_type:
            raise ConflictError("resource TTL write is pending for a different object type")
        # A crash between registry publication and metadata publication must
        # not replace the pending generation with an unmanaged new object.
        fields = {"expires_at": pending.expires_at, "ttl_generation": pending.generation}
        await write_resource_fields(fs, object_type, uri, fields, ctx=ctx, lease_ref=lease_ref)
        return fields
    parent_uri = uri.rsplit("/", 1)[0]
    parent_fields = await resource_ttl_fields(fs, parent_uri, ctx=ctx)
    if parent_fields and not resource_ttl:
        # A document's children share its lifecycle; do not register each chunk.
        return parent_fields
    manager = getattr(fs, "runtime_config_manager", None)
    config = None
    if manager is not None:
        from openviking.config.merge import apply_three_state_patch
        from openviking_cli.utils.config.ttl_config import TTLConfig

        # Reuse sparse runtime overrides: changing one directory must not reset
        # other directories or stop inheriting the library's global policy.
        def resolve(view):
            override = view.account.ttl
            return TTLConfig.model_validate(
                apply_three_state_patch(
                    view.cluster.ttl.model_dump(by_alias=True),
                    override.model_dump(by_alias=True, exclude_unset=True) if override else {},
                )
            )

        config = await manager.resolve_account(ctx.account_id, resolve)
    fields = None if existing else freeze_ttl_fields(uri, resource_ttl=resource_ttl, config=config)
    if fields is None:
        return {}
    await write_resource_fields(fs, object_type, uri, fields, ctx=ctx, lease_ref=lease_ref)
    return fields


async def write_resource_fields(fs, object_type, uri, fields, *, ctx, lease_ref):
    metadata_uri = ttl_metadata_uri(object_type, uri)
    path = fs._uri_to_path(metadata_uri, ctx=ctx)
    metadata_lease = await fs._async_agfs.pathlock_acquire_exact(path, owner_lease_ref=lease_ref)
    try:
        await fs.write_file(metadata_uri, json.dumps(fields), ctx=ctx, lease_ref=metadata_lease)
    finally:
        await fs._async_agfs.pathlock_release(metadata_lease)


async def update_resource_expiry(fs, uri: str, expires_at: str, *, ctx) -> dict:
    """Explicitly revise a live resource's expiry under its existing object lock."""
    from openviking.storage.acl import AclAction

    if ttl_scope_for_uri(uri) != "resources":
        raise InvalidArgumentError("uri must identify a resource")
    try:
        expiry = format_iso8601(parse_iso_datetime(expires_at))
    except (ValueError, TypeError) as exc:
        raise InvalidArgumentError("expires_at must be an ISO 8601 timestamp") from exc
    if hidden_by_ttl(expiry):
        raise InvalidArgumentError("expires_at must be in the future")
    await fs._ensure_access(uri, ctx, action=AclAction.WRITE)
    stat = await fs.stat(uri, ctx=ctx)
    is_dir = bool(stat.get("isDir"))
    object_type = OBJECT_TYPE_RESOURCE if is_dir else OBJECT_TYPE_RESOURCE_FILE
    path = fs._uri_to_path(uri, ctx=ctx)
    lock = fs._async_agfs.pathlock_acquire_tree if is_dir else fs._async_agfs.pathlock_acquire_exact
    lease = await lock(path)
    try:
        await fs.stat(uri, ctx=ctx)
        fields = await read_resource_fields(fs, object_type, uri, ctx=ctx)
        if not fields or not fields.get("expires_at"):
            raise InvalidArgumentError("resource has no frozen TTL to update")
        if hidden_by_ttl(fields["expires_at"]):
            raise NotFoundError(uri, "resource")
        fields["expires_at"] = expiry
        await write_resource_fields(fs, object_type, uri, fields, ctx=ctx, lease_ref=lease)
        return {"uri": uri, **fields}
    finally:
        await fs._async_agfs.pathlock_release(lease)


async def resource_ttl_snapshot(fs, uris, *, ctx):
    """Use the same source snapshots for semantic generation and its final fence."""
    if not uris or ttl_scope_for_uri(uris[0]) != "resources":
        return None
    if not ttl_enabled() and not await fs.ttl_registry.account_may_have_records(ctx.account_id):
        return None
    snapshot = {}
    for uri in uris:
        fields = await resource_ttl_fields(fs, uri, ctx=ctx)
        snapshot[uri] = (
            await resource_ttl_visible(fs, uri, ctx=ctx, require_source=True),
            fields.get("ttl_generation"),
            fields.get("expires_at"),
        )
    return snapshot
