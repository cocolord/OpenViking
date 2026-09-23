# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""TTL aliases retain sparse runtime PATCH semantics across config reloads."""

from types import SimpleNamespace

import pytest

from openviking.config.binding import manager_over_source
from openviking.config.scope import ConfigScope
from openviking.config.source import MemoryConfigSource
from openviking.config.ttl import resolve_ttl_config
from openviking.config.validate import ConfigPatchError, normalize_config_keys, validate_patch
from openviking_cli.utils.config import TTLConfig, get_openviking_config, set_openviking_config
from openviking_cli.utils.config.open_viking_config import OpenVikingConfig


@pytest.mark.asyncio
async def test_alias_and_field_name_patch_update_one_value_and_null_restores_baseline():
    original = get_openviking_config()
    base = original.model_copy(
        update={"ttl": TTLConfig(global_default={"mode": "days", "ttl_days": 30})}
    )
    source = MemoryConfigSource()
    manager = manager_over_source(source, base_config=base)
    fs = SimpleNamespace(runtime_config_manager=manager)
    try:
        await manager.initialize()
        await manager.patch_cluster({"ttl": {"global": {"mode": "days", "ttl_days": 20}}})
        await manager.patch_cluster({"ttl": {"global_default": {"ttl_days": 15}}})
        assert (await resolve_ttl_config(fs, "acct")).global_default.ttl_days == 15
        await manager.patch_account(
            "acct", {"ttl": {"global_default": {"mode": "days", "ttl_days": 7}}}
        )
        await manager.patch_account("acct", {"ttl": {"global": {"ttl_days": 5}}})
        assert (await resolve_ttl_config(fs, "acct")).global_default.ttl_days == 5
        reloaded = manager_over_source(source, base_config=base)
        await reloaded.initialize()
        assert (
            await resolve_ttl_config(SimpleNamespace(runtime_config_manager=reloaded), "acct")
        ).global_default.ttl_days == 5
        assert (await reloaded.get_settings(ConfigScope.account("acct")))["ttl"] == {
            "global": {"mode": "days", "ttl_days": 5}
        }
        await manager.patch_account("acct", {"ttl": {"global_default": None}})
        assert (await resolve_ttl_config(fs, "acct")).global_default.ttl_days == 15
        await manager.patch_cluster({"ttl": {"global_default": None}})
        assert (await resolve_ttl_config(fs, "acct")).global_default.ttl_days == 30
    finally:
        set_openviking_config(original)


def test_alias_does_not_bypass_runtime_write_restrictions():
    with pytest.raises(ConfigPatchError):
        validate_patch(OpenVikingConfig, {"storage": {"vectordb": {"project": "changed"}}})
    with pytest.raises(ConfigPatchError, match="either"):
        normalize_config_keys(OpenVikingConfig, {"ttl": {"global": None, "global_default": None}})
