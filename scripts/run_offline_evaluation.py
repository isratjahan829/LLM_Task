#!/usr/bin/env python
"""Offline evaluation harness for LLM RAG pipelines using PDF ground truth."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.evaluation import (
    MetricsComputer,
    MetricsConfig,
    RAGEvaluator,
    load_eval_dataset,
    load_model_responses,
)
from src.evaluation.plotting import DEFAULT_METRICS, plot_metric_bars


LOGGER = logging.getLogger("rag_evaluation")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to JSONL file containing evaluation questions and reference answers.",
    )
    parser.add_argument(
        "--responses",
        action="append",
        required=True,
        help=(
            "Model response specification in the form path:model_name[:trial]. "
            "Add multiple --responses flags for additional models or trials."
        ),
    )
    parser.add_argument(
        "--summary-output",
        default="reports/summary_metrics.csv",
        help="Where to write aggregated metrics (CSV).",
    )
    parser.add_argument(
        "--samples-output",
        default="reports/sample_metrics.csv",
        help="Where to write per-sample metrics (CSV).",
    )
    parser.add_argument(
        "--summary-json",
        default="reports/summary_metrics.json",
        help="Where to write aggregated metrics (JSON).",
    )
    parser.add_argument(
        "--plot-output",
        default="plots/model_metric_comparison.png",
        help="Path for the generated comparison bar chart.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Subset of metrics to plot; defaults to latency, cosine, F1, BERTScore, completeness, hallucination, irrelevance, meteor, bleu, avg_response_length.",
    )
    parser.add_argument(
        "--embedding-model",
        default=MetricsConfig.embedding_model,
        help="Sentence-Transformer model to use for cosine similarity.",
    )
    parser.add_argument(
        "--bert-score-model",
        default=MetricsConfig.bert_score_model,
        help="Model name to use for BERTScore computations.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (cpu, cuda, cuda:0, mps) for embedding/BERTScore models.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    dataset = load_eval_dataset(args.dataset)
    LOGGER.info("Loaded %d evaluation samples", len(dataset))

    response_specs = [parse_response_spec(spec) for spec in args.responses]

    all_responses = []
    for spec in response_specs:
        LOGGER.info(
            "Loading responses for %s (trial %d) from %s",
            spec.model_name,
            spec.trial,
            spec.path,
        )
        responses = load_model_responses(spec.path, spec.model_name, default_trial=spec.trial)
        all_responses.extend(responses)

    metrics_config = MetricsConfig(
        embedding_model=args.embedding_model,
        bert_score_model=args.bert_score_model,
        device=args.device,
    )
    metrics_computer = MetricsComputer(metrics_config)
    evaluator = RAGEvaluator(dataset, metrics_computer)

    result = evaluator.evaluate(all_responses)

    write_dataframe(result.sample_metrics, args.samples_output)
    write_dataframe(result.summary_metrics, args.summary_output)
    write_json(result.summary_metrics, args.summary_json)

    metrics_to_plot = args.metrics if args.metrics else DEFAULT_METRICS
    plot_metric_bars(
        result.summary_metrics,
        metrics=metrics_to_plot,
        output_path=args.plot_output,
        title="Model Metric Comparison",
    )

    LOGGER.info("Evaluation complete. Summary written to %s", args.summary_output)
    LOGGER.info("Comparison plot saved to %s", args.plot_output)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ResponseSpec:
    def __init__(self, path: Path, model_name: str, trial: int) -> None:
        self.path = path
        self.model_name = model_name
        self.trial = trial


def parse_response_spec(spec: str) -> ResponseSpec:
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(
            f"Invalid response spec '{spec}'. Expected format path:model_name[:trial]"
        )

    path = Path(parts[0])
    model_name = parts[1]
    if len(parts) == 3:
        try:
            trial = int(parts[2])
        except ValueError as exc:
            raise ValueError(f"Trial must be an integer in '{spec}'") from exc
    else:
        trial = 1

    return ResponseSpec(path=path, model_name=model_name, trial=trial)


def write_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def write_json(df: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = df.to_dict(orient="records")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


if __name__ == "__main__":
    main()
