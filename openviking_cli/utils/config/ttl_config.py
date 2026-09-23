# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""TTL (time-to-live) policy configuration for events and sessions.

TTL is default OFF and only ever applies to three supported directory scopes:

- ``user_events``  -> ``viking://user/{user_id}/memories/events/``
- ``peer_events``  -> ``viking://user/{user_id}/peers/{peer_id}/memories/events/``
- ``sessions``     -> ``viking://user/{user_id}/sessions/``

It is not extended to preferences, resources, entities, or any other directory.

The configuration mirrors the design doc's minimal ``policy`` protocol: a library
global default, per-scope defaults, and optional concrete-directory overrides.
All levels use the same ``TTLPolicy`` structure. Resolution order is *nearest
directory override > scope default > library global default > off*.
"""

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# The three supported TTL scopes. Deliberately closed: TTL never applies to any
# other directory type.
TTLScope = Literal["user_events", "peer_events", "sessions"]
TTL_SCOPES: tuple[TTLScope, ...] = ("user_events", "peer_events", "sessions")


def _is_supported_directory_uri(uri: str) -> bool:
    """Accept directory policies, never an individual event or session."""
    if not uri.startswith("viking://") or "?" in uri or "#" in uri:
        return False
    path = uri[len("viking://") :].rstrip("/")
    parts = path.split("/")
    if any(not part for part in parts) or len(parts) < 3 or parts[0] != "user":
        return False
    if parts[2] == "sessions":
        # A session ID is an object root, not a configurable directory.
        return len(parts) == 3
    if len(parts) >= 4 and parts[2:4] == ["memories", "events"]:
        event_path = parts[4:]
    elif (
        len(parts) >= 6
        and parts[2] == "peers"
        and parts[4:6]
        == [
            "memories",
            "events",
        ]
    ):
        event_path = parts[6:]
    else:
        return False
    # Event objects are Markdown files; accepting one as a policy key would
    # silently reintroduce the per-object TTL excluded by the product review.
    return not any(part.endswith(".md") for part in event_path)


class TTLPolicy(BaseModel):
    """A single TTL policy node shared by the global default and each scope.

    - ``inherit``: defer to the next explicit ancestor, then scope -> global ->
      off. Valid for directories and scope defaults, never for the library global.
    - ``disabled``: explicitly no TTL; blocks inheritance from the global level.
    - ``days``: expire ``ttl_days`` after object creation. ``ttl_days`` is then a
      required positive integer (minimum 1 day). "Off" is expressed with
      ``disabled``, never with ``0`` or a negative value.
    """

    mode: Literal["inherit", "disabled", "days"] = "inherit"
    ttl_days: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_ttl_days(self) -> "TTLPolicy":
        if self.mode == "days":
            if self.ttl_days is None:
                raise ValueError(
                    "ttl_days is required and must be a positive integer when mode='days'"
                )
        elif self.ttl_days is not None:
            raise ValueError(f"ttl_days must be omitted when mode='{self.mode}'")
        return self


class TTLConfig(BaseModel):
    """Cluster-wide TTL policy. Default OFF.

    The default instance leaves the global policy ``disabled`` and every scope
    ``inherit``, so nothing expires unless an operator opts in. Changing this
    config only affects objects created afterwards — existing objects keep the
    ``ttl_days`` snapshot frozen at their creation time.
    """

    global_default: TTLPolicy = Field(
        default_factory=lambda: TTLPolicy(mode="disabled"),
        alias="global",
        description=(
            "Library-global TTL default. Only 'disabled' or 'days' are allowed here; "
            "'inherit' has nothing above it to inherit from."
        ),
    )
    user_events: TTLPolicy = Field(
        default_factory=TTLPolicy,
        description="TTL default for viking://user/{user_id}/memories/events/.",
    )
    peer_events: TTLPolicy = Field(
        default_factory=TTLPolicy,
        description="TTL default for viking://user/{user_id}/peers/{peer_id}/memories/events/.",
    )
    sessions: TTLPolicy = Field(
        default_factory=TTLPolicy,
        description="TTL default for viking://user/{user_id}/sessions/.",
    )
    directories: Dict[str, TTLPolicy] = Field(
        default_factory=dict,
        description=(
            "TTL overrides keyed by a concrete in-scope Viking directory URI. "
            "The nearest matching ancestor wins."
        ),
    )

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _check_global(self) -> "TTLConfig":
        if self.global_default.mode == "inherit":
            raise ValueError(
                "ttl.global mode must be 'disabled' or 'days' ('inherit' is not "
                "allowed at the global level)"
            )
        normalized: Dict[str, TTLPolicy] = {}
        for raw_uri, policy in self.directories.items():
            uri = raw_uri.rstrip("/")
            if not _is_supported_directory_uri(uri):
                raise ValueError(
                    "ttl.directories keys must be concrete directories under "
                    "user events, peer events, or sessions"
                )
            if uri in normalized:
                raise ValueError(f"duplicate ttl directory after normalization: {uri}")
            normalized[uri] = policy
        self.directories = normalized
        return self

    def resolve_scope(self, scope: TTLScope) -> Optional[int]:
        """Return the effective ``ttl_days`` for a scope, or ``None`` when off.

        Resolution order: the scope's own policy first; ``inherit`` falls through
        to the global default; ``disabled`` blocks inheritance and yields off.
        """
        policy = getattr(self, scope)
        if policy.mode == "days":
            return policy.ttl_days
        if policy.mode == "disabled":
            return None
        # mode == "inherit": fall through to the global default.
        if self.global_default.mode == "days":
            return self.global_default.ttl_days
        return None

    def resolve_uri(self, uri: str, scope: TTLScope) -> Optional[int]:
        """Resolve one in-scope URI using its nearest directory override.

        URI scope validation intentionally stays in :mod:`openviking.core.ttl`;
        this config model only performs boundary-safe ancestor matching. This
        keeps configuration parsing independent from server-side URI modules.
        """
        normalized_uri = uri.rstrip("/")
        matches = (
            (directory, policy)
            for directory, policy in self.directories.items()
            if normalized_uri == directory or normalized_uri.startswith(directory + "/")
        )
        for _, policy in sorted(matches, key=lambda item: len(item[0]), reverse=True):
            if policy.mode == "days":
                return policy.ttl_days
            if policy.mode == "disabled":
                return None
            # Inherit the next explicit ancestor before falling back to scope.
        return self.resolve_scope(scope)

    @property
    def enabled(self) -> bool:
        """True when TTL resolves to an active expiry for at least one scope."""
        return any(self.resolve_scope(scope) is not None for scope in TTL_SCOPES) or any(
            policy.mode == "days" for policy in self.directories.values()
        )
