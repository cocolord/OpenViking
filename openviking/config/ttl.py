# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Resolve the same library TTL policy for every object creation path."""

from openviking.config.merge import apply_three_state_patch
from openviking_cli.utils.config.ttl_config import TTLConfig


async def resolve_ttl_config(fs, account_id: str) -> TTLConfig | None:
    manager = getattr(fs, "runtime_config_manager", None)
    if manager is None:
        return None  # The pure TTL helpers fall back to ov.conf at startup.

    def resolve(view):
        override = view.account.ttl
        return TTLConfig.model_validate(
            apply_three_state_patch(
                view.cluster.ttl.model_dump(by_alias=True),
                override.model_dump(by_alias=True, exclude_unset=True) if override else {},
            )
        )

    return await manager.resolve_account(account_id, resolve)
