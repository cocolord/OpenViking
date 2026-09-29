# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from datetime import datetime, timezone

import pytest

from openviking.utils.time_decay import (
    build_time_decay_fusion_spec,
    build_time_decay_post_process_ops,
    parse_duration_ms,
    time_decay_candidate_limit,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0),
        ("0m", 0),
        ("15m", 900_000),
        ("2h", 7_200_000),
        ("3d", 259_200_000),
        ("1095000d", 94_608_000_000_000),
    ],
)
def test_parse_duration_ms(value, expected):
    assert parse_duration_ms(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "-1d", "1.5h", "1s", "1D", " 1d", "1095001d", "99999999999999999999d", 1, None],
)
def test_parse_duration_ms_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_duration_ms(value)


def test_enabled_decay_uses_the_server_owned_curve():
    origin = datetime(2026, 1, 8, tzinfo=timezone.utc)
    spec = build_time_decay_fusion_spec(protection="1d", origin=origin)

    assert spec.field == "updated_at"
    assert spec.origin_ms == origin.timestamp() * 1000
    assert spec.offset_ms == 24 * 3600 * 1000
    assert spec.scale_ms == 7 * 24 * 3600 * 1000
    assert spec.decay == 0.5


@pytest.mark.parametrize("protection", ["0", "0m", "0h", "0d"])
def test_builder_accepts_zero_protection_duration(protection):
    spec = build_time_decay_fusion_spec(
        protection=protection,
        origin=datetime(2026, 1, 8, tzinfo=timezone.utc),
    )

    assert spec.offset_ms == 0


@pytest.mark.parametrize("protection", ["0", "0m", "0h", "0d"])
def test_post_process_omits_zero_offset(protection):
    origin = datetime(2026, 1, 8, tzinfo=timezone.utc)

    ops = build_time_decay_post_process_ops(protection=protection, origin=origin)

    addition = ops[0]["addition_score"][0]
    assert ops[0]["fusion_by"] == "multiply"
    assert "addition_score_weight" not in ops[0]
    assert "offset" not in addition
    assert addition == {
        "factor": 1,
        "base_value_from": "decay_func",
        "field": "updated_at",
        "func": "exp",
        "origin": "2026-01-08T00:00:00.000Z",
        "scale": "7d",
        "decay": 0.5,
    }


def test_time_decay_candidate_budget_is_bounded():
    assert time_decay_candidate_limit(10) == 30
    assert time_decay_candidate_limit(10, offset=5) == 45
    assert time_decay_candidate_limit(100_000) == 100_000
    with pytest.raises(ValueError, match="must not exceed 100000"):
        time_decay_candidate_limit(100_001)
