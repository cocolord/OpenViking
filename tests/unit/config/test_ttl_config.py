# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Unit tests for TTL policy configuration (``ttl_config``).

Assert the default is OFF, the ``inherit``/``disabled``/``days`` resolution
order, and the validators that keep the policy well-formed (positive ttl_days,
no ``inherit`` at the global level).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from openviking_cli.utils.config.ttl_config import (
    TTL_SCOPES,
    TTLConfig,
    TTLPolicy,
)


def test_default_config_is_off():
    config = TTLConfig()
    assert config.enabled is False
    for scope in TTL_SCOPES:
        assert config.resolve_uri_policy("", scope).mode == "disabled"


def test_scope_disabled_blocks_global_inheritance():
    config = TTLConfig(
        **{"global": {"mode": "days", "ttl_days": 7}},
        sessions={"mode": "disabled"},
    )
    assert config.resolve_uri_policy("", "sessions").mode == "disabled"
    assert config.resolve_uri_policy("", "user_events") == TTLPolicy(mode="days", ttl_days=7)


def test_global_inherit_is_rejected():
    with pytest.raises(ValidationError):
        TTLConfig(**{"global": {"mode": "inherit"}})


@pytest.mark.parametrize(
    "policy",
    [
        {"mode": "days"},
        {"mode": "days", "ttl_days": 0},
        {"mode": "days", "ttl_days": -5},
        {"mode": "disabled", "ttl_days": 5},
        {"mode": "inherit", "ttl_days": 5},
    ],
)
def test_invalid_days_policy_is_rejected(policy):
    with pytest.raises(ValidationError):
        TTLPolicy(**policy)


def test_directory_only_policy_enables_ttl_and_normalizes_slash():
    config = TTLConfig(directories={"viking://user/u1/sessions/": {"mode": "days", "ttl_days": 2}})
    assert config.enabled is True
    assert "viking://user/u1/sessions" in config.directories


@pytest.mark.parametrize(
    "uri",
    [
        "/local/a",
        "viking://resources/docs",
        "viking://user/u1/peers/p1/memories/events",
        "viking://user/u1/peers/p1/memories/events/",
        "viking://user/u1/preferences",
        "viking://user/u1/memories/entities",
        "viking://user/u1/sessions/s1",
        "viking://user/u1/memories/events/../entities",
        "viking://user/u1/memories/events//bad",
        "viking://user/u1/memories/events/notes.md",
        "viking://user/u1/memories/events/2026",
        "viking://user/u1/memories/events/2026/09",
        "viking://user/u1/memories/events/2026/09/30",
        "viking://user/u1/memories/events/2026/09/30/a.md",
    ],
)
def test_only_concrete_policy_roots_are_configurable(uri):
    with pytest.raises(ValidationError):
        TTLConfig(directories={uri: {"mode": "days", "ttl_days": 7}})


def test_resources_are_rejected():
    with pytest.raises(ValidationError):
        TTLConfig(resources={"mode": "days", "ttl_days": 7})


def test_session_defaults_accept_absolute_retention():
    for value in (
        {"sessions": {"mode": "absolute", "ttl_absolute": 2000000000}},
        {
            "directories": {
                "viking://user/u1/sessions": {"mode": "absolute", "ttl_absolute": 2000000000}
            }
        },
    ):
        config = TTLConfig.model_validate(value)
        assert (
            config.resolve_uri_policy("viking://user/u1/sessions/s1", "sessions").mode == "absolute"
        )


@pytest.mark.parametrize(
    "root_policy",
    [
        TTLPolicy(mode="days", ttl_days=14),
        TTLPolicy(mode="absolute", ttl_absolute=2000000000),
        TTLPolicy(mode="disabled"),
        TTLPolicy(mode="inherit"),
    ],
)
def test_root_policy_overrides_type_then_library_default(root_policy):
    config = TTLConfig.model_validate(
        {
            "global": {"mode": "days", "ttl_days": 7},
            "user_events": {"mode": "days", "ttl_days": 30},
            "peer_events": {"mode": "days", "ttl_days": 21},
            "directories": {"viking://user/u1/memories/events": root_policy},
        }
    )
    assert config.enabled is True
    for peer in ("p1", "p2"):
        assert config.resolve_uri_policy(
            f"viking://user/u1/peers/{peer}/memories/events/2026/10/09/a.md", "peer_events"
        ) == TTLPolicy(mode="days", ttl_days=21)
    assert config.resolve_uri_policy(
        "viking://user/u1/memories/events/2026/09/30/a.md", "user_events"
    ) == (config.user_events if root_policy.mode == "inherit" else root_policy)
    assert config.resolve_uri_policy(
        "viking://user/u2/memories/events/2026/09/30/a.md", "user_events"
    ) == TTLPolicy(mode="days", ttl_days=30)
    assert config.resolve_uri_policy("viking://user/u2/sessions/s1", "sessions") == TTLPolicy(
        mode="days", ttl_days=7
    )


@pytest.mark.parametrize("days", [1, 30, 365000])
def test_days_boundary_values_remain_days(days):
    assert TTLPolicy(mode="days", ttl_days=days).model_dump()["ttl_days"] == days


@pytest.mark.parametrize("days", [0, 365001])
def test_days_outside_supported_range_are_rejected(days):
    with pytest.raises(ValidationError):
        TTLPolicy(mode="days", ttl_days=days)
