"""Evaluation orchestration for model responses over a PDF-derived dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .dataset import EvalSample, ModelResponse
from .metrics import MetricsComputer


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EvaluationResult:
    """Container for evaluation outputs."""

    sample_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame


class RAGEvaluator:
    """Evaluate RAG model outputs using a suite of automatic metrics."""

    def __init__(
        self,
        samples: Dict[str, EvalSample],
        metrics_computer: MetricsComputer,
        *,
        drop_missing: bool = False,
    ) -> None:
        self.samples = samples
        self.metrics_computer = metrics_computer
        self.drop_missing = drop_missing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(self, responses: Iterable[ModelResponse]) -> EvaluationResult:
        rows = []
        skipped = 0

        for response in responses:
            sample = self.samples.get(response.sample_id)
            if sample is None:
                skipped += 1
                LOGGER.warning("Skipping response with unknown sample id: %s", response.sample_id)
                continue

            metrics = self.metrics_computer.compute_all(
                reference_answer=sample.reference_answer,
                prediction=response.answer,
                question=sample.question,
                reference_context=sample.reference_context,
                retrieved_context=response.retrieved_context,
            )

            row = {
                "model": response.model_name,
                "trial": response.trial,
                "sample_id": response.sample_id,
                "latency": response.latency,
                **metrics,
            }
            rows.append(row)

        if not rows:
            raise ValueError("No evaluation rows were produced; check dataset and responses")

        sample_metrics = pd.DataFrame(rows)

        if self.drop_missing:
            sample_metrics = sample_metrics.dropna()

        summary_metrics = self._summarize(sample_metrics)

        if skipped:
            LOGGER.info("%d responses skipped due to missing sample ids", skipped)

        return EvaluationResult(sample_metrics=sample_metrics, summary_metrics=summary_metrics)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _summarize(self, sample_metrics: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = [
            "latency",
            "precision",
            "recall",
            "f1",
            "cosine_similarity",
            "bleu",
            "meteor",
            "bert_score_f1",
            "completeness",
            "hallucination",
            "irrelevance",
            "response_length",
            "response_char_length",
        ]

        summary = (
            sample_metrics
            .groupby(["model", "trial"], as_index=False)[numeric_columns]
            .mean(numeric_only=True)
        )

        summary = self._augment_summary(summary)
        return summary

    def _augment_summary(self, summary: pd.DataFrame) -> pd.DataFrame:
        summary = summary.copy()

        summary.rename(
            columns={
                "response_length": "avg_response_length",
                "response_char_length": "avg_response_char_length",
            },
            inplace=True,
        )

        # Higher latency is worse. Convert to a score in [0, 1].
        summary["latency_score"] = summary["latency"].apply(self._latency_to_score)

        positive_metrics = [
            "f1",
            "bert_score_f1",
            "completeness",
            "meteor",
            "bleu",
            "cosine_similarity",
        ]
        negative_metrics = ["hallucination", "irrelevance"]

        pos_values = summary[positive_metrics].mean(axis=1, skipna=True)
        neg_values = (1.0 - summary[negative_metrics]).mean(axis=1, skipna=True)

        # Combine into a single quality score for easy ranking.
        summary["quality_score"] = (
            (
                pos_values * len(positive_metrics)
                + neg_values * len(negative_metrics)
            )
            / (len(positive_metrics) + len(negative_metrics))
        )

        # Ensure values remain within bounds.
        bounded_columns = [
            "latency_score",
            "quality_score",
            "f1",
            "bert_score_f1",
            "completeness",
            "hallucination",
            "irrelevance",
            "meteor",
            "bleu",
            "cosine_similarity",
        ]
        for column in bounded_columns:
            summary[column] = summary[column].apply(self._clip_unit_interval)

        return summary

    @staticmethod
    def _latency_to_score(value: Optional[float]) -> float:
        if value is None or np.isnan(value) or value < 0:
            return 0.0
        return 1.0 / (1.0 + value)

    @staticmethod
    def _clip_unit_interval(value: float) -> float:
        if value is None or np.isnan(value):
            return 0.0
        return float(np.clip(value, 0.0, 1.0))
