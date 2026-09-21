# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""VikingDB-compatible event time-decay score fusion helpers.

The product contract blends the semantic and time scores directly. Keeping
that contract in one module lets local collections and VikingDB requests use
the same curve and ranking semantics.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime, timezone
from numbers import Real
from typing import Any, Iterable, Mapping, Optional, Sequence

from openviking.utils.time_utils import format_iso8601, parse_iso_datetime

DEFAULT_POST_PROCESS_FACTOR = 3
MAX_VECTOR_POST_PROCESS_INPUT_LIMIT = 100_000
# This curve is an internal ranking contract validated against VikingDB's
# score_fusion operator; it is intentionally not part of ov.conf/ovcli.conf.
EVENT_TIME_DECAY_SCALE = "7d"
EVENT_TIME_DECAY_DECAY = 0.5
# Match VikingDB's date_time duration limit: 3000 years of 365 days.
MAX_DURATION_MS = 3000 * 365 * 24 * 60 * 60 * 1000

_DURATION_RE = re.compile(r"^(0|[0-9]+[mhd])$")
_DURATION_MULTIPLIERS_MS = {
    "m": 60 * 1000,
    "h": 60 * 60 * 1000,
    "d": 24 * 60 * 60 * 1000,
}


def validate_time_decay_weight(value: Any) -> float:
    """Validate the public event time-decay weight."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("events_time_decay_weight must be a finite number in [0, 1)")
    weight = float(value)
    if not math.isfinite(weight) or weight < 0.0 or weight >= 1.0:
        raise ValueError("events_time_decay_weight must be a finite number in [0, 1)")
    return weight


def _blend_scores(origin_score: float, addition_score: float, weight: float) -> float:
    return (1.0 - weight) * origin_score + weight * addition_score


def parse_duration_ms(value: Any, *, parameter_name: str = "duration") -> int:
    """Parse ``0`` or a non-negative integer duration with m/h/d units."""
    if not isinstance(value, str) or not _DURATION_RE.fullmatch(value):
        raise ValueError(
            f"{parameter_name} must be '0' or a non-negative integer followed by m, h, or d"
        )
    if value == "0":
        return 0
    duration_ms = int(value[:-1]) * _DURATION_MULTIPLIERS_MS[value[-1]]
    if duration_ms > MAX_DURATION_MS:
        raise ValueError(f"{parameter_name} exceeds the maximum duration of 1095000d")
    return duration_ms


def _datetime_to_epoch_ms(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("time value must not be boolean")
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("time value must be finite")
        return result
    if isinstance(value, str):
        value = parse_iso_datetime(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp() * 1000.0
    raise ValueError("time value must be an epoch millisecond number or ISO 8601 string")


@dataclass(frozen=True)
class TimeDecayFusionSpec:
    """Compiled score-fusion parameters for event time decay."""

    weight: float
    field: str
    origin_ms: float
    offset_ms: int
    scale_ms: int
    decay: float
    factor: float = 1.0
    _decay_rate: float = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Validate and compile request constants once, outside the candidate loop.
        validate_time_decay_weight(self.weight)
        if self.scale_ms <= 0:
            raise ValueError("time-decay scale must be greater than zero")
        if not 0.0 < self.decay < 1.0:
            raise ValueError("time-decay decay must be in (0, 1)")
        object.__setattr__(self, "origin_ms", _datetime_to_epoch_ms(self.origin_ms))
        object.__setattr__(self, "_decay_rate", math.log(self.decay) / self.scale_ms)

    def fuse(self, origin_score: float, source_time: Any) -> tuple[float, float]:
        """Return ``(final_score, raw_addition_score)`` for one candidate."""
        distance_ms = max(
            0.0, abs(_datetime_to_epoch_ms(source_time) - self.origin_ms) - self.offset_ms
        )
        addition_score = self.factor * math.exp(self._decay_rate * distance_ms)
        final_score = _blend_scores(origin_score, addition_score, self.weight)
        return final_score, addition_score

    def fuse_optional(self, origin_score: float, source_time: Any) -> tuple[float, Optional[float]]:
        """Fuse only reliable source times; otherwise preserve the origin score."""
        try:
            return self.fuse(origin_score, source_time)
        except (TypeError, ValueError, OverflowError):
            return origin_score, None


def fuse_time_decay_scores(*, origin_score: float, addition_score: float, weight: float) -> float:
    """Blend semantic and time scores directly, as defined by the PRD."""
    checked_weight = validate_time_decay_weight(weight)
    return _blend_scores(origin_score, addition_score, checked_weight)


def apply_time_decay_to_search_items(
    items: Iterable[Any],
    spec: TimeDecayFusionSpec,
    *,
    source_fields: Optional[Iterable[Mapping[str, Any]]] = None,
) -> list[Any]:
    """Apply local score fusion to SearchItemResult-like objects and stable-sort them."""
    processed = list(items)
    fields_iter: list[Optional[Mapping[str, Any]]] = (
        list(source_fields) if source_fields is not None else [None] * len(processed)
    )
    for item, raw_fields in zip(processed, fields_iter, strict=False):
        origin_score = float(item.score or 0.0)
        fields = raw_fields if isinstance(raw_fields, Mapping) else item.fields
        if not isinstance(fields, Mapping):
            fields = {}
        item.score, item.addition_score = spec.fuse_optional(origin_score, fields.get(spec.field))
        item.origin_score = origin_score
    processed.sort(key=lambda item: item.score or 0.0, reverse=True)
    return processed


def build_time_decay_post_process_ops(
    *,
    weight: float,
    protection: str,
    origin: Optional[datetime] = None,
    field: str = "updated_at",
) -> list[dict[str, Any]]:
    """Build the VikingDB score-fusion payload; zero weight emits no operator."""
    checked_weight = validate_time_decay_weight(weight)
    if checked_weight == 0.0:
        return []
    protection_ms = parse_duration_ms(protection, parameter_name="events_time_decay_protection")
    request_time = origin or datetime.now(timezone.utc)
    addition = {
        "factor": 1,
        "base_value_from": "decay_func",
        "field": field,
        "func": "exp",
        "origin": format_iso8601(request_time),
        "scale": EVENT_TIME_DECAY_SCALE,
        "decay": EVENT_TIME_DECAY_DECAY,
    }
    # VikingDB documents offset as optional with a zero default, while its
    # date_time parser rejects explicit zero durations (both "0" and "0d").
    if protection_ms > 0:
        addition["offset"] = protection
    return [
        {
            "op": "score_fusion",
            "fusion_by": "add",
            "addition_score_weight": checked_weight,
            "normalize_for_origin_score": {"enable": False},
            "normalize_for_addition_score": {"enable": False},
            "addition_score": [addition],
        }
    ]


def parse_time_decay_post_process_ops(
    post_process_ops: Optional[Sequence[Mapping[str, Any]]],
) -> Optional[TimeDecayFusionSpec]:
    """Parse the score-fusion subset emitted by this module for local use."""
    if not post_process_ops:
        return None
    if len(post_process_ops) != 1:
        raise ValueError("local vector search supports one score_fusion operator")

    op = post_process_ops[0]
    if op.get("op") != "score_fusion" or op.get("fusion_by") != "add":
        raise ValueError("local vector search only supports additive score_fusion")
    disabled_normalization = {"enable": False}
    for key in ("normalize_for_origin_score", "normalize_for_addition_score"):
        if op.get(key) != disabled_normalization:
            raise ValueError("local time-decay fusion requires score normalization to be disabled")

    weight = validate_time_decay_weight(op.get("addition_score_weight"))
    if weight == 0.0:
        raise ValueError("score_fusion addition_score_weight must be in (0, 1)")
    additions = op.get("addition_score")
    if not isinstance(additions, list) or len(additions) != 1:
        raise ValueError("local time-decay fusion requires exactly one addition_score item")
    addition = additions[0]
    if not isinstance(addition, Mapping):
        raise ValueError("addition_score item must be an object")
    if addition.get("base_value_from") != "decay_func" or addition.get("func") != "exp":
        raise ValueError("local time-decay fusion requires an exponential decay_func")

    scale_ms = parse_duration_ms(addition.get("scale"), parameter_name="time-decay scale")
    if "decay" not in addition:
        raise ValueError("local time-decay fusion requires an explicit decay")
    decay = float(addition["decay"])
    factor = float(addition.get("factor", 1.0))
    if not math.isfinite(factor) or factor == 0.0:
        raise ValueError("time-decay factor must be finite and non-zero")

    return TimeDecayFusionSpec(
        weight=weight,
        field=str(addition.get("field", "")),
        origin_ms=_datetime_to_epoch_ms(addition.get("origin", datetime.now(timezone.utc))),
        offset_ms=parse_duration_ms(
            addition.get("offset", "0"), parameter_name="time-decay offset"
        ),
        scale_ms=scale_ms,
        decay=decay,
        factor=factor,
    )


def post_process_input_limit(limit: int, offset: int = 0) -> int:
    """Return VikingDB's default 3x candidate budget for vector search."""
    final_window = limit + offset
    if final_window > MAX_VECTOR_POST_PROCESS_INPUT_LIMIT:
        raise ValueError(
            "time-decay search limit + offset must not exceed "
            f"{MAX_VECTOR_POST_PROCESS_INPUT_LIMIT}"
        )
    return min(
        final_window * DEFAULT_POST_PROCESS_FACTOR,
        MAX_VECTOR_POST_PROCESS_INPUT_LIMIT,
    )
