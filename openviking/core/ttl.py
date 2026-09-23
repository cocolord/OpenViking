# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Central TTL resolution: map an object URI to its expiry.

This is the single seam that turns a canonical Viking URI plus the cluster TTL
config into a frozen ``expires_at`` at object-creation time. Every writer that
freezes TTL (events via the memory path, sessions via SessionMeta) and the
background cleanup scanner go through here so the scope rules stay in one place.

TTL is default OFF and strictly scoped to three directory kinds:

- ``user_events``  -> ``viking://user/{uid}/memories/events/...``
- ``peer_events``  -> ``viking://user/{uid}/peers/{pid}/memories/events/...``
- ``sessions``     -> ``viking://user/{uid}/sessions/{sid}...``

Day granularity is expressed as ``ttl_days`` whole days after ``received_at``
(N x 24h in UTC). ``expires_at`` is authoritative for both the read barrier and
the cleanup scan.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional
from uuid import uuid4

from openviking.core.namespace import uri_parts
from openviking.storage.internal_names import WEBDAV_RESERVED_FILENAMES, is_storage_internal_name
from openviking.utils.time_utils import format_iso8601, parse_iso_datetime
from openviking_cli.utils.config import TTLConfig, TTLScope, get_openviking_config

# Object-type tags used by lifecycle records / cleanup, kept next to the scope
# rules so callers do not re-derive them.
OBJECT_TYPE_EVENT = "event"
OBJECT_TYPE_SESSION = "session"
TTL_GENERATION_FIELD = "ttl_generation"
TTL_FIELD_NAMES = frozenset({"ttl_days", "received_at", "expires_at", TTL_GENERATION_FIELD})


def ttl_scope_for_uri(uri: str) -> Optional[TTLScope]:
    """Classify a canonical URI into a TTL scope, or ``None`` when unscoped.

    Only user events, peer events, and sessions are in scope. Anything else
    (preferences, resources, entities, skills, non-event memories, ...) returns
    ``None`` so TTL never touches it.
    """
    try:
        parts = uri_parts(uri)
    except ValueError:
        return None
    if len(parts) < 3 or parts[0] != "user":
        return None
    # sessions: viking://user/{uid}/sessions/...
    if parts[2] == "sessions":
        return "sessions"
    # peer events: viking://user/{uid}/peers/{pid}/memories/events/...
    if len(parts) >= 6 and parts[2] == "peers" and parts[4] == "memories" and parts[5] == "events":
        return "peer_events"
    # user events: viking://user/{uid}/memories/events/...
    if len(parts) >= 4 and parts[2] == "memories" and parts[3] == "events":
        return "user_events"
    return None


def ttl_object_for_uri(uri: str, *, is_dir: bool = False) -> Optional[tuple[str, str]]:
    """Return ``(object_type, canonical_object_uri)`` for a TTL object path.

    A session's root metadata controls its complete subtree. Event files may
    have any extension (or none), just like public content writes. Callers
    walking the filesystem must identify directories with ``is_dir``; event
    containers and reserved system files are not independently expiring objects.
    User-authored dot-files follow the same TTL rules as other event files.
    """
    scope = ttl_scope_for_uri(uri)
    if scope is None:
        return None
    try:
        parts = uri_parts(uri)
    except ValueError:
        return None
    if scope == "sessions":
        if len(parts) < 4:
            return None
        return OBJECT_TYPE_SESSION, "viking://" + "/".join(parts[:4])
    event_root_depth = 6 if scope == "peer_events" else 4
    if (
        is_dir
        or len(parts) <= event_root_depth
        or parts[-1] in WEBDAV_RESERVED_FILENAMES
        or is_storage_internal_name(parts[-1])
    ):
        return None
    return OBJECT_TYPE_EVENT, "viking://" + "/".join(parts)


def resolve_ttl_days(uri: str, config: Optional[TTLConfig] = None) -> Optional[int]:
    """Resolve the effective ``ttl_days`` for a URI, or ``None`` when TTL is off."""
    scope = ttl_scope_for_uri(uri)
    if scope is None:
        return None
    ttl_config = config if config is not None else _current_ttl_config()
    if ttl_config is None:
        return None
    return ttl_config.resolve_uri(uri, scope)


def compute_expires_at(received_at: datetime, ttl_days: int) -> datetime:
    """Return the frozen expiry: ``received_at`` plus ``ttl_days`` whole days."""
    if received_at.tzinfo is None:
        received_at = received_at.replace(tzinfo=timezone.utc)
    return received_at + timedelta(days=ttl_days)


def freeze_ttl_fields(
    uri: str,
    *,
    received_at: Optional[datetime] = None,
    config: Optional[TTLConfig] = None,
) -> Optional[dict]:
    """Compute the frozen TTL snapshot for a new object, or ``None`` when off.

    Returns a dict with RFC 3339 ``received_at``/``expires_at`` strings and the
    integer ``ttl_days`` actually applied. Callers persist this snapshot verbatim
    at creation time and never recompute it from later config changes.
    """
    ttl_days = resolve_ttl_days(uri, config)
    if ttl_days is None:
        return None
    received = received_at or datetime.now(timezone.utc)
    if received.tzinfo is None:
        received = received.replace(tzinfo=timezone.utc)
    expires = compute_expires_at(received, ttl_days)
    return {
        "ttl_days": ttl_days,
        "received_at": format_iso8601(received),
        "expires_at": format_iso8601(expires),
        # An incarnation fence, not a policy field.  A URI delete/recreate gets
        # a new value so delayed cleanup/embedding work cannot touch the new
        # object.  Ordinary updates and session renewal preserve it.
        TTL_GENERATION_FIELD: str(uuid4()),
    }


def apply_ttl_fields(
    uri: str,
    metadata: Mapping[str, Any],
    *,
    existing_fields: Optional[Mapping[str, Any]] = None,
    received_at: Optional[datetime] = None,
    config: Optional[TTLConfig] = None,
) -> dict[str, Any]:
    """Return metadata with system-owned TTL fields frozen or preserved.

    On creation (``existing_fields is None``), caller-provided TTL fields are
    discarded and a snapshot is derived from the effective policy. On update,
    the existing object's fields are copied verbatim. This prevents public and
    LLM write paths from choosing or changing expiry independently.
    """
    result = {key: value for key, value in metadata.items() if key not in TTL_FIELD_NAMES}
    if existing_fields is None:
        snapshot = freeze_ttl_fields(uri, received_at=received_at, config=config)
        if snapshot:
            result.update(snapshot)
        return result
    for field in TTL_FIELD_NAMES:
        value = existing_fields.get(field)
        if value is not None and value != "":
            result[field] = value
    return result


def is_expired(expires_at: Optional[str], *, now: Optional[datetime] = None) -> bool:
    """Return whether an ``expires_at`` timestamp is at or past ``now`` (UTC).

    Absent/blank/unparseable expiry means "no TTL" and is never expired, matching
    the read barrier's absent-field-visible rule.
    """
    if not expires_at:
        return False
    try:
        expires = parse_iso_datetime(expires_at)
    except Exception:
        return False
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return expires <= current


def ttl_enabled() -> bool:
    """Whether current policy creates TTL snapshots for new objects.

    This switch is deliberately not consulted by visibility checks. Policy
    changes only affect objects created afterwards; an object that already has
    a frozen ``expires_at`` must not become visible again when policy is later
    disabled.
    """
    config = _current_ttl_config()
    return config is not None and config.enabled


def hidden_by_ttl(expires_at: Optional[str], *, now: Optional[datetime] = None) -> bool:
    """Whether a read/compute path should treat ``expires_at`` as logically gone.

    Used by filesystem reads and vector candidate validation against source
    metadata. Visibility follows the frozen object snapshot, not
    current policy: disabling TTL stops new snapshots but cannot revive an
    already-expired object. Objects without ``expires_at`` remain visible.
    """
    return is_expired(expires_at, now=now)


def _current_ttl_config() -> Optional[TTLConfig]:
    try:
        return get_openviking_config().ttl
    except Exception:
        # Config not initialized (e.g. unit tests, bootstrap). Fail closed to OFF.
        return None
