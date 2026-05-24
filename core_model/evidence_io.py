from __future__ import annotations

import json
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from .evidence_corpus import EvidenceCorpus
from .evidence_schema import (
    EvidenceQualityTier,
    EvidenceType,
    EvidenceUnit,
    OutcomeLabel,
    OversightLabel,
    UserExpertise,
)
from .source_registry import SourceRegistry, SourceStatus, SourceType


class EvidenceLoadError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        path: str | Path,
        line_number: int | None = None,
    ) -> None:
        self.message = message
        self.path = Path(path)
        self.line_number = line_number
        super().__init__(str(self))

    def __str__(self) -> str:
        location = str(self.path)
        if self.line_number is not None:
            location = f"{location}: line {self.line_number}"
        return f"{location}: {self.message}"


def _load_json_array(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvidenceLoadError(str(exc), path=path) from exc
    except JSONDecodeError as exc:
        raise EvidenceLoadError(exc.msg, path=path, line_number=exc.lineno) from exc

    if not isinstance(loaded, list):
        raise EvidenceLoadError("expected a JSON array", path=path)
    for index, entry in enumerate(loaded, start=1):
        if not isinstance(entry, dict):
            raise EvidenceLoadError(
                f"expected source entry {index} to be an object",
                path=path,
            )
    return loaded


def _coerce_enum(
    enum_type: type[Enum],
    value: Any,
    field_name: str,
    *,
    path: str | Path,
    line_number: int | None = None,
) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError as exc:
        raise EvidenceLoadError(
            f"invalid {field_name}: {value!r}",
            path=path,
            line_number=line_number,
        ) from exc


def _add_source_entry(
    registry: SourceRegistry,
    entry: dict[str, Any],
    *,
    path: str | Path,
) -> None:
    try:
        source = registry.add_source(
            source_id=entry["source_id"],
            source_name=entry["source_name"],
            source_type=_coerce_enum(
                SourceType, entry["source_type"], "source_type", path=path
            ),
            owner_or_publisher=entry["owner_or_publisher"],
            license_or_access=entry["license_or_access"],
            update_cadence=entry["update_cadence"],
            coverage=entry["coverage"],
            known_biases=entry["known_biases"],
            quality_tier=_coerce_enum(
                EvidenceQualityTier,
                entry["quality_tier"],
                "quality_tier",
                path=path,
            ),
        )
    except KeyError as exc:
        raise EvidenceLoadError(f"missing source field: {exc.args[0]}", path=path) from exc
    except ValueError as exc:
        raise EvidenceLoadError(str(exc), path=path) from exc

    if "status" in entry:
        status = _coerce_enum(SourceStatus, entry["status"], "status", path=path)
        if status is not source.status or entry.get("status_reason"):
            try:
                registry.update_status(
                    source.source_id,
                    status,
                    reason=entry.get("status_reason"),
                )
            except ValueError as exc:
                raise EvidenceLoadError(str(exc), path=path) from exc


def load_source_registry(path: str | Path) -> SourceRegistry:
    registry = SourceRegistry()
    for entry in _load_json_array(path):
        _add_source_entry(registry, entry, path=path)
    return registry
