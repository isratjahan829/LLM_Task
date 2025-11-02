"""Utilities for loading evaluation datasets and model responses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class DatasetValidationError(RuntimeError):
    """Raised when the evaluation dataset or responses are malformed."""


@dataclass(slots=True)
class EvalSample:
    """Container for a single evaluation sample."""

    sample_id: str
    question: str
    reference_answer: str
    reference_context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelResponse:
    """Container for a single model response associated with an evaluation sample."""

    sample_id: str
    model_name: str
    answer: str
    trial: int = 1
    latency: Optional[float] = None
    retrieved_context: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"Invalid JSON on line {line_number} in {path}: {exc}"
                ) from exc


def load_eval_dataset(path: str | Path) -> Dict[str, EvalSample]:
    """Load evaluation samples from a JSONL file.

    Each line should contain ``id``, ``question`` and ``reference_answer`` fields. Optional
    fields include ``reference_context`` and ``metadata``.
    """

    dataset_path = Path(path)
    samples: Dict[str, EvalSample] = {}

    for record in _load_jsonl(dataset_path):
        sample_id = record.get("id") or record.get("sample_id")
        if not sample_id:
            raise DatasetValidationError(
                f"Missing `id` field in dataset record from {dataset_path}"
            )

        question = record.get("question")
        reference_answer = record.get("reference_answer")
        if not question or not reference_answer:
            raise DatasetValidationError(
                f"Record {sample_id} missing required fields `question` or `reference_answer`"
            )

        samples[sample_id] = EvalSample(
            sample_id=sample_id,
            question=question,
            reference_answer=reference_answer,
            reference_context=record.get("reference_context"),
            metadata=record.get("metadata", {}),
        )

    if not samples:
        raise DatasetValidationError(f"No samples loaded from {dataset_path}")

    return samples


def load_model_responses(
    path: str | Path,
    model_name: str,
    default_trial: int = 1,
) -> List[ModelResponse]:
    """Load model responses from a JSONL file.

    Each line should contain ``id`` (or ``sample_id``) and ``answer``. Optional fields include
    ``latency`` (seconds), ``trial`` (int), ``retrieved_context`` and arbitrary metadata.
    """

    response_path = Path(path)
    responses: List[ModelResponse] = []

    for record in _load_jsonl(response_path):
        sample_id = record.get("id") or record.get("sample_id")
        answer = record.get("answer") or record.get("response")

        if not sample_id or answer is None:
            raise DatasetValidationError(
                f"Invalid response record in {response_path}; `id` and `answer` are required"
            )

        responses.append(
            ModelResponse(
                sample_id=sample_id,
                model_name=model_name,
                answer=answer,
                trial=int(record.get("trial", default_trial)),
                latency=_safe_float(record.get("latency")),
                retrieved_context=record.get("retrieved_context"),
                raw={k: v for k, v in record.items() if k not in {"id", "sample_id", "answer"}},
            )
        )

    if not responses:
        raise DatasetValidationError(f"No responses loaded from {response_path}")

    return responses


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
