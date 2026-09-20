# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UpsertDataResult:
    ids: List[Any] = field(default_factory=list)


@dataclass
class UpdateResult:
    ok: bool = False
    ids: List[str] = field(default_factory=list)
    updated_count: int = 0
    error_code: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class DataItem:
    id: Any = None
    fields: Optional[Dict[str, Any]] = None


@dataclass
class FetchDataInCollectionResult:
    items: List[DataItem] = field(default_factory=list)
    ids_not_exist: List[Any] = field(default_factory=list)


@dataclass
class SearchItemResult:
    id: Any = None
    fields: Optional[Dict[str, Any]] = None
    score: Optional[float] = None
    origin_score: Optional[float] = None
    addition_score: Optional[float] = None


@dataclass
class SearchResult:
    data: List[SearchItemResult] = field(default_factory=list)


def parse_remote_search_result(
    payload: Dict[str, Any],
    *,
    post_process_ops: Optional[Sequence[Mapping[str, Any]]] = None,
) -> SearchResult:
    """Parse VikingDB results and reconstruct requested time-decay scores."""
    from openviking.utils.time_decay import parse_time_decay_post_process_ops

    decay_spec = None
    if post_process_ops:
        try:
            decay_spec = parse_time_decay_post_process_ops(post_process_ops)
        except (TypeError, ValueError):
            # Collection APIs can carry post-process operators other than the
            # OpenViking time-decay subset. They still benefit from ann_score
            # compatibility, but must not be interpreted as time decay.
            pass

    result = SearchResult()
    if not isinstance(payload, dict) or "data" not in payload:
        return result

    for item in payload.get("data", []):
        origin_score = item.get("origin_score")
        addition_score = item.get("addition_score")
        if post_process_ops and origin_score is None:
            origin_score = item.get("ann_score")

        if decay_spec is not None and origin_score is not None and addition_score is None:
            fields = item.get("fields")
            source_time = fields.get(decay_spec.field) if isinstance(fields, Mapping) else None
            _, locally_computed_score = decay_spec.fuse_optional(float(origin_score), source_time)
            # The unnormalized additive formula is reversible. Prefer the
            # cloud-derived value so explanation scores reproduce its returned
            # final score exactly, including server-side numeric precision. A
            # missing or invalid timestamp remains explicitly unexplained.
            if locally_computed_score is not None and item.get("score") is not None:
                addition_score = (
                    float(item["score"]) - (1.0 - decay_spec.weight) * float(origin_score)
                ) / decay_spec.weight
            else:
                addition_score = locally_computed_score

        result.data.append(
            SearchItemResult(
                id=item.get("id"),
                fields=item.get("fields"),
                score=item.get("score"),
                origin_score=origin_score,
                addition_score=addition_score,
            )
        )
    return result


@dataclass
class AggregateResult:
    """Result of aggregation operation.

    Attributes:
        agg: Aggregation result dictionary
             - Total count: {"_total": <count>}
             - Grouped count: {"value1": count1, "value2": count2, ...}
        op: Aggregation operation name (e.g., "count")
        field: Field name used for grouping (None for total count)
    """

    agg: Dict[str, Any] = field(default_factory=dict)
    op: str = "count"
    field: Optional[str] = None
