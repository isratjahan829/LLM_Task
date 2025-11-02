"""Visualization utilities for RAG evaluation reports."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_METRICS = [
    "latency",
    "cosine_similarity",
    "f1",
    "bert_score_f1",
    "completeness",
    "hallucination",
    "irrelevance",
    "meteor",
    "bleu",
    "avg_response_length",
]


def plot_metric_bars(
    summary_df: pd.DataFrame,
    *,
    metrics: Sequence[str] | None = None,
    output_path: str | Path,
    title: str | None = None,
    height: float = 4.0,
    aspect: float = 1.2,
) -> Path:
    """Create a facet bar-chart comparing models across metrics."""

    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("summary_df must be a pandas DataFrame")

    if metrics is None:
        metrics = DEFAULT_METRICS

    for metric in metrics:
        if metric not in summary_df.columns:
            raise ValueError(f"Metric '{metric}' not found in summary dataframe columns")

    plot_df = summary_df.melt(
        id_vars=["model", "trial"],
        value_vars=list(metrics),
        var_name="metric",
        value_name="score",
    )

    plot_df.sort_values(["metric", "model", "trial"], inplace=True)

    sns.set_theme(style="whitegrid")

    unique_metrics = list(dict.fromkeys(plot_df["metric"].tolist()))
    col_wrap = 3 if len(unique_metrics) > 3 else len(unique_metrics)

    g = sns.catplot(
        data=plot_df,
        x="model",
        y="score",
        hue="trial",
        col="metric",
        kind="bar",
        height=height,
        aspect=aspect,
        col_wrap=col_wrap,
        sharey=False,
    )

    g.set_axis_labels("Model", "Score")
    g.set_titles(col_template="{col_name}")
    if title:
        g.fig.suptitle(title, fontsize=16)
        g.fig.subplots_adjust(top=0.88)

    for ax in g.axes.flatten():
        for container in ax.containers:
            ax.bar_label(container, fmt="{:.2f}", fontsize=8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(output_path, bbox_inches="tight")
    plt.close(g.fig)
    return output_path
