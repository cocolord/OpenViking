# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from openviking.storage.index_action import IndexAction
from openviking.utils.tags import merge_search_tags, normalize_search_tags

_UPDATE_FIELD_ALLOWLIST = frozenset(
    {"md5", "content", "abstract", "updated_at", "active_count", "tags", "search_tags"}
)
_FIELD_MODES = frozenset({"replace", "append"})


class IncompleteInitialRecordError(RuntimeError):
    """A missing record cannot be recreated from the queued initial value."""

    def __init__(self, record_id: str, missing_fields: List[str]):
        self.record_id = record_id
        self.missing_fields = list(missing_fields)
        super().__init__(
            "update_fields could not recreate missing vector record: "
            f"record_id={record_id} missing_fields={self.missing_fields}"
        )


def apply_field_patch(
    base: Dict[str, Any],
    incoming: Dict[str, Any],
    field_modes: Dict[str, str],
) -> Dict[str, Any]:
    """Apply a scalar patch without treating omitted fields as empty values."""

    result = dict(base)
    for field_name, value in incoming.items():
        mode = field_modes.get(field_name, "replace")
        if field_name == "search_tags":
            normalized = normalize_search_tags(value, discard_invalid=True)
            result[field_name] = (
                merge_search_tags(result.get(field_name), normalized)
                if mode == "append"
                else normalized
            )
            continue
        if mode != "replace":
            raise ValueError(f"field {field_name} does not support mode {mode}")
        result[field_name] = value
    return result


def missing_initial_record_fields(record: Dict[str, Any]) -> List[str]:
    """Return fields required to create a usable vector record."""

    missing = [field for field in ("id", "uri", "account_id", "level") if record.get(field) is None]
    if not record.get("vector") and not record.get("sparse_vector"):
        missing.append("vector")
    return missing


@dataclass
class EmbeddingMsg:
    """Durable embedding-queue message for model or index-only work.

    ``context_data`` holds context identity and normal scalar data for
    ``UPSERT``/``MERGE``. ``record_ids``/``update_fields`` make delete and scalar
    updates explicit rather than encoding them as special embedding payloads.
    """

    message: Optional[Union[str, List[Dict[str, Any]]]]
    context_data: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid4()))
    telemetry_id: str = ""
    action: IndexAction = IndexAction.UPSERT
    record_ids: List[str] = field(default_factory=list)
    update_fields: Dict[str, Any] = field(default_factory=dict)
    field_modes: Dict[str, str] = field(default_factory=dict)
    initial_fields: Dict[str, Any] = field(default_factory=dict)
    queue_enqueued_at: float = 0.0

    def __init__(
        self,
        message: Optional[Union[str, List[Dict[str, Any]]]],
        context_data: Dict[str, Any],
        telemetry_id: str = "",
        action: IndexAction | str = IndexAction.UPSERT,
        record_ids: Optional[List[str]] = None,
        update_fields: Optional[Dict[str, Any]] = None,
        field_modes: Optional[Dict[str, str]] = None,
        initial_fields: Optional[Dict[str, Any]] = None,
        queue_enqueued_at: float = 0.0,
    ):
        self.id = str(uuid4())
        self.message = message
        self.context_data = context_data
        self.telemetry_id = telemetry_id
        self.action = IndexAction(action)
        self.record_ids = list(record_ids or [])
        self.update_fields = dict(update_fields or {})
        self.field_modes = {str(key): str(value) for key, value in (field_modes or {}).items()}
        self.initial_fields = dict(initial_fields or {})
        self.queue_enqueued_at = max(float(queue_enqueued_at or 0.0), 0.0)
        self._validate()

    def _validate(self) -> None:
        if self.action is IndexAction.NONE:
            if (
                self.message is not None
                or self.record_ids
                or self.update_fields
                or self.field_modes
                or self.initial_fields
            ):
                raise ValueError("none action cannot carry work")
            return
        if self.action in {IndexAction.UPSERT, IndexAction.MERGE}:
            if not isinstance(self.message, (str, list)):
                raise ValueError(f"{self.action.value} requires an embedding message")
            if self.action is IndexAction.UPSERT and self.field_modes:
                raise ValueError("upsert carries resolved fields and cannot use field modes")
            self._validate_field_patch()
            return
        if self.message is not None:
            raise ValueError(f"{self.action.value} does not accept an embedding message")
        if not self.record_ids:
            raise ValueError(f"{self.action.value} requires record_ids")
        if self.action is IndexAction.DELETE:
            if self.update_fields or self.field_modes or self.initial_fields:
                raise ValueError("delete does not accept update fields or initial fields")
            return
        if len(self.record_ids) != 1:
            raise ValueError("update_fields requires exactly one record id")
        unknown = set(self.update_fields) - _UPDATE_FIELD_ALLOWLIST
        if unknown:
            raise ValueError(f"update_fields contains forbidden fields: {sorted(unknown)}")
        if not self.update_fields:
            raise ValueError("update_fields must not be empty")
        self._validate_field_patch()

    def _validate_field_patch(self) -> None:
        unknown = set(self.update_fields) - _UPDATE_FIELD_ALLOWLIST
        if unknown:
            raise ValueError(f"update_fields contains forbidden fields: {sorted(unknown)}")
        unknown_modes = set(self.field_modes) - set(self.update_fields)
        if unknown_modes:
            raise ValueError(f"field modes without update fields: {sorted(unknown_modes)}")
        invalid_modes = {
            field: mode for field, mode in self.field_modes.items() if mode not in _FIELD_MODES
        }
        if invalid_modes:
            raise ValueError(f"invalid field modes: {invalid_modes}")
        unsupported_append = {
            field for field, mode in self.field_modes.items() if mode == "append" and field != "search_tags"
        }
        if unsupported_append:
            raise ValueError(f"append is unsupported for fields: {sorted(unsupported_append)}")

    @classmethod
    def for_update_fields(
        cls,
        *,
        record_id: str,
        fields: Dict[str, Any],
        context_data: Dict[str, Any],
        field_modes: Optional[Dict[str, str]] = None,
        initial_fields: Optional[Dict[str, Any]] = None,
        telemetry_id: str = "",
    ) -> "EmbeddingMsg":
        return cls(
            message=None,
            context_data=context_data,
            telemetry_id=telemetry_id,
            action=IndexAction.UPDATE_FIELDS,
            record_ids=[record_id],
            update_fields=fields,
            field_modes=field_modes,
            initial_fields=initial_fields,
        )

    @classmethod
    def for_delete(
        cls,
        *,
        record_ids: List[str],
        context_data: Dict[str, Any],
        telemetry_id: str = "",
    ) -> "EmbeddingMsg":
        return cls(
            message=None,
            context_data=context_data,
            telemetry_id=telemetry_id,
            action=IndexAction.DELETE,
            record_ids=record_ids,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert embedding message to dictionary format."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert embedding message to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingMsg":
        """Create an embedding message object from dictionary."""
        obj = EmbeddingMsg(
            message=data["message"],
            context_data=data["context_data"],
            telemetry_id=data.get("telemetry_id", ""),
            action=data.get("action", IndexAction.UPSERT.value),
            record_ids=data.get("record_ids"),
            update_fields=data.get("update_fields"),
            field_modes=data.get("field_modes"),
            initial_fields=data.get("initial_fields"),
            queue_enqueued_at=data.get("queue_enqueued_at", 0.0),
        )
        obj.id = data.get("id", obj.id)
        return obj

    @classmethod
    def from_json(cls, json_str: str) -> "EmbeddingMsg":
        """Safely create object from JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")
