# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Shared index actions used by plans and embedding queue messages."""

from enum import Enum


class IndexAction(str, Enum):
    """Action applied to one or more vector-index records."""

    NONE = "none"
    UPSERT = "upsert"
    MERGE = "merge"
    UPDATE_FIELDS = "update_fields"
    DELETE = "delete"
