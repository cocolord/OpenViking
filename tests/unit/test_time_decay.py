# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from datetime import datetime, timezone

import pytest

from openviking.storage.vectordb.collection.result import SearchItemResult
from openviking.utils.time_decay import (
    TimeDecayFusionSpec,
    build_time_decay_post_process_ops,
    exponential_decay_score,
    fuse_time_decay_scores,
    parse_duration_ms,
    parse_time_decay_post_process_ops,
    post_process_input_limit,
    validate_time_decay_weight,
)
from openviking_cli.utils.config import RetrievalConfig


def _candidate_set_delta(reference, comparison):
    reference_ids = [item.id for item in reference]
    comparison_ids = [item.id for item in comparison]
    return {
        "reference_ids": reference_ids,
        "comparison_ids": comparison_ids,
        "only_in_reference": sorted(set(reference_ids) - set(comparison_ids)),
        "only_in_comparison": sorted(set(comparison_ids) - set(reference_ids)),
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0", 0), ("0m", 0), ("15m", 900_000), ("2h", 7_200_000), ("3d", 259_200_000)],
)
def test_parse_duration_ms(value, expected):
    assert parse_duration_ms(value) == expected


@pytest.mark.parametrize("value", ["", "-1d", "1.5h", "1s", "1D", " 1d", 1, None])
def test_parse_duration_ms_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_duration_ms(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.0, True, "0.5"])
def test_validate_time_decay_weight_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        validate_time_decay_weight(value)


def test_exponential_decay_honors_protection_and_scale():
    origin = datetime(2026, 1, 8, tzinfo=timezone.utc)

    protected = exponential_decay_score(
        datetime(2026, 1, 7, 12, tzinfo=timezone.utc),
        origin=origin,
        offset_ms=parse_duration_ms("1d"),
        scale_ms=parse_duration_ms("7d"),
        decay=0.5,
    )
    one_scale_after_protection = exponential_decay_score(
        datetime(2025, 12, 31, tzinfo=timezone.utc),
        origin=origin,
        offset_ms=parse_duration_ms("1d"),
        scale_ms=parse_duration_ms("7d"),
        decay=0.5,
    )

    assert protected == pytest.approx(1.0)
    assert one_scale_after_protection == pytest.approx(0.5)


def test_builder_and_local_parser_share_cloud_contract():
    origin = datetime(2026, 1, 8, tzinfo=timezone.utc)
    ops = build_time_decay_post_process_ops(
        weight=0.2, protection="1d", origin=origin, scale="7d", decay=0.5
    )

    assert ops == [
        {
            "op": "score_fusion",
            "fusion_by": "add",
            "addition_score_weight": 0.2,
            "normalize_for_origin_score": {"enable": False},
            "normalize_for_addition_score": {"enable": False},
            "addition_score": [
                {
                    "factor": 1,
                    "base_value_from": "decay_func",
                    "field": "updated_at",
                    "func": "exp",
                    "origin": "2026-01-08T00:00:00.000Z",
                    "offset": "1d",
                    "scale": "7d",
                    "decay": 0.5,
                }
            ],
        }
    ]

    spec = parse_time_decay_post_process_ops(ops)
    assert spec is not None
    fused, addition = spec.fuse(0.8, "2025-12-31T00:00:00.000Z")
    assert addition == pytest.approx(0.5)
    assert fused == pytest.approx(0.8 * 0.8 + 0.2 * 0.5)


@pytest.mark.parametrize("protection", ["0", "0m", "0h", "0d"])
def test_builder_omits_zero_offset_for_cloud_date_time_duration(protection):
    ops = build_time_decay_post_process_ops(
        weight=0.2,
        protection=protection,
        origin=datetime(2026, 1, 8, tzinfo=timezone.utc),
        scale="7d",
        decay=0.5,
    )

    assert "offset" not in ops[0]["addition_score"][0]


def test_zero_weight_keeps_post_processing_disabled():
    assert build_time_decay_post_process_ops(weight=0.0, protection="0") == []


def test_enabled_decay_requires_an_explicit_internal_curve():
    with pytest.raises(ValueError, match="curve is not configured"):
        build_time_decay_post_process_ops(weight=0.2, protection="0")


def test_direct_score_fusion_matches_prd_formula():
    assert fuse_time_decay_scores(
        origin_score=0.8, addition_score=0.5, weight=0.2
    ) == pytest.approx(0.74)


def test_internal_curve_config_has_no_guessed_default_and_requires_a_pair():
    config = RetrievalConfig()
    assert config.events_time_decay_scale is None
    assert config.events_time_decay_decay is None

    with pytest.raises(ValueError, match="must be configured together"):
        RetrievalConfig(events_time_decay_scale="7d")
    with pytest.raises(ValueError, match="must be configured together"):
        RetrievalConfig(events_time_decay_decay=0.5)

    configured = RetrievalConfig(events_time_decay_scale="30d", events_time_decay_decay=0.5)
    assert configured.events_time_decay_scale == "30d"
    assert configured.events_time_decay_decay == 0.5


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


def test_post_process_candidate_budget_matches_cloud_default():
    assert post_process_input_limit(10) == 30
    assert post_process_input_limit(10, offset=5) == 45
    assert post_process_input_limit(100_000) == 100_000


def test_candidate_set_delta_reports_differences_in_result_order():
    baseline = [SearchItemResult(id=value) for value in ["a", "b", "c"]]
    decayed = [SearchItemResult(id=value) for value in ["c", "d", "a"]]

    assert _candidate_set_delta(baseline, decayed) == {
        "reference_ids": ["a", "b", "c"],
        "comparison_ids": ["c", "d", "a"],
        "only_in_reference": ["b"],
        "only_in_comparison": ["d"],
    }
