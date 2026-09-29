# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from datetime import datetime, timezone

import pytest

from openviking.utils.time_decay import (
    TimeDecayFusionSpec,
    build_time_decay_fusion_spec,
    build_time_decay_post_process_ops,
    parse_duration_ms,
    rank_time_decay_scores,
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

    factors = spec.time_scores(["2025-12-31T00:00:00.000Z", "2026-01-07T12:00:00.000Z"])
    assert factors == pytest.approx([0.5, 1.0])


@pytest.mark.parametrize("source_time", ["2026-01-01T00:00:00Z", "2026-01-15T00:00:00Z"])
def test_time_distance_matches_cloud_decay(source_time):
    spec = build_time_decay_fusion_spec(
        protection="0",
        origin=datetime(2026, 1, 8, tzinfo=timezone.utc),
    )

    assert spec.time_scores([source_time]) == pytest.approx([0.5])


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


def test_native_score_fusion_ranks_mixed_candidates_and_keeps_stable_ties():
    ranked = rank_time_decay_scores([0.8, 0.6, 0.4, 0.9], [0.5, None, 1.0, 0.0], 3)
    assert [index for index, _ in ranked] == [1, 0, 2]
    assert [score for _, score in ranked] == pytest.approx([0.6, 0.4, 0.4])


@pytest.mark.parametrize("scores,factors,limit", [([], [], 10), ([0.8], [0.5], 0)])
def test_native_ranking_empty_result(scores, factors, limit):
    assert rank_time_decay_scores(scores, factors, limit) == []


@pytest.mark.parametrize(
    "scores,factors,limit",
    [([0.8], [], 1), ([float("nan")], [1.0], 1), ([0.8], [1.5], 1), ([0.8], [0.5], -1)],
)
def test_native_ranking_rejects_invalid_inputs(scores, factors, limit):
    with pytest.raises(ValueError):
        rank_time_decay_scores(scores, factors, limit)


@pytest.mark.parametrize("source_time", [None, "", "not-a-time", float("nan")])
def test_missing_or_invalid_time_keeps_origin_score(source_time):
    spec = TimeDecayFusionSpec(
        field="updated_at",
        origin_ms=1_000.0,
        offset_ms=0,
        scale_ms=1_000,
        decay=0.5,
    )

    factors = spec.time_scores([source_time])
    assert factors == [None]
    assert rank_time_decay_scores([0.42], factors, 1) == [(0, 0.42)]


def test_time_decay_candidate_budget_is_bounded():
    assert time_decay_candidate_limit(10) == 30
    assert time_decay_candidate_limit(10, offset=5) == 45
    assert time_decay_candidate_limit(100_000) == 100_000
    with pytest.raises(ValueError, match="must not exceed 100000"):
        time_decay_candidate_limit(100_001)
