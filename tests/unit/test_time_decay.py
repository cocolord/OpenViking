# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from datetime import datetime, timezone

import pytest

from openviking.utils.time_decay import (
    TimeDecayFusionSpec,
    build_time_decay_fusion_spec,
    fuse_time_decay_scores,
    parse_duration_ms,
    time_decay_candidate_limit,
    validate_time_decay_weight,
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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.0, True, "0.5"])
def test_validate_time_decay_weight_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        validate_time_decay_weight(value)


def test_enabled_decay_uses_the_server_owned_curve():
    origin = datetime(2026, 1, 8, tzinfo=timezone.utc)
    spec = build_time_decay_fusion_spec(weight=0.2, protection="1d", origin=origin)

    fused, addition = spec.fuse(0.8, "2025-12-31T00:00:00.000Z")
    assert addition == pytest.approx(0.5)
    assert fused == pytest.approx(0.8 * 0.8 + 0.2 * 0.5)


@pytest.mark.parametrize("protection", ["0", "0m", "0h", "0d"])
def test_builder_accepts_zero_protection_duration(protection):
    spec = build_time_decay_fusion_spec(
        weight=0.2,
        protection=protection,
        origin=datetime(2026, 1, 8, tzinfo=timezone.utc),
    )

    assert spec.offset_ms == 0


def test_direct_score_fusion_matches_prd_formula():
    assert fuse_time_decay_scores(
        origin_score=0.8, addition_score=0.5, weight=0.2
    ) == pytest.approx(0.74)


@pytest.mark.parametrize("source_time", [None, "", "not-a-time", float("nan")])
def test_missing_or_invalid_time_keeps_origin_score(source_time):
    spec = TimeDecayFusionSpec(
        weight=0.8,
        field="updated_at",
        origin_ms=1_000.0,
        offset_ms=0,
        scale_ms=1_000,
        decay=0.5,
    )

    final_score, time_score = spec.fuse_optional(0.42, source_time)

    assert final_score == pytest.approx(0.42)
    assert time_score is None


def test_time_decay_candidate_budget_is_bounded():
    assert time_decay_candidate_limit(10) == 30
    assert time_decay_candidate_limit(10, offset=5) == 45
    assert time_decay_candidate_limit(100_000) == 100_000
    with pytest.raises(ValueError, match="must not exceed 100000"):
        time_decay_candidate_limit(100_001)
