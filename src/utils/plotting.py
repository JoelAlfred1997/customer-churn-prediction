"""Shared plotting utilities for consistent figure styling and export."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PALETTE: list[str] = ["#2563EB", "#DC2626", "#6B7280", "#16A34A", "#D97706"]
CHURN_PALETTE: dict[str, str] = {"No": "#2563EB", "Yes": "#DC2626"}
FIG_DPI: int = 150


def set_plot_style() -> None:
    """Apply project-wide matplotlib/seaborn theme.

    Call once at the top of each notebook or script before creating figures.
    """
    sns.set_theme(
        style="whitegrid",
        palette=PALETTE,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": FIG_DPI,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        },
    )


def save_fig(
    fig: plt.Figure,
    name: str,
    figures_dir: Path | str = "reports/figures",
) -> Path:
    """Save *fig* as a PNG under *figures_dir*.

    Args:
        fig: Matplotlib figure to save.
        name: Filename stem (no extension).
        figures_dir: Output directory; created if absent.

    Returns:
        Resolved path of the saved file.
    """
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    return path.resolve()


def plot_histogram(
    series: pd.Series,
    *,
    ax: plt.Axes | None = None,
    bins: int = 30,
    color: str = PALETTE[0],
    title: str | None = None,
    xlabel: str | None = None,
    show_mean: bool = True,
) -> plt.Axes:
    """Draw a styled histogram for a numeric series.

    Args:
        series: Numeric data; NaN values are dropped before plotting.
        ax: Axes to draw on; creates a new figure if None.
        bins: Number of histogram bins.
        color: Bar fill colour.
        title: Axes title; defaults to series.name.
        xlabel: x-axis label; defaults to series.name.
        show_mean: Overlay a dashed vertical line at the mean.

    Returns:
        The axes used.
    """
    if ax is None:
        _, ax = plt.subplots()

    clean = series.dropna()
    ax.hist(clean, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

    if show_mean:
        mean_val = float(clean.mean())
        ax.axvline(
            mean_val,
            color="#111827",
            linestyle="--",
            linewidth=1.2,
            label=f"mean={mean_val:.1f}",
        )
        ax.legend(frameon=False)

    ax.set_title(title or series.name or "")
    ax.set_xlabel(xlabel or series.name or "")
    ax.set_ylabel("Count")
    return ax


def plot_bar(
    counts: pd.Series,
    *,
    ax: plt.Axes | None = None,
    color: str | list[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    pct: bool = True,
    horizontal: bool = False,
) -> plt.Axes:
    """Draw a styled bar chart from a value_counts-style Series.

    Args:
        counts: Series mapping category label → count.
        ax: Axes to draw on; creates a new figure if None.
        color: Single colour or per-bar list; defaults to project palette.
        title: Axes title.
        xlabel: Axis label (x for vertical bars, y for horizontal).
        pct: Annotate each bar with its percentage of the total.
        horizontal: Render horizontal bars (barh).

    Returns:
        The axes used.
    """
    if ax is None:
        _, ax = plt.subplots()

    bar_color: str | list[str] = color if color is not None else PALETTE[: len(counts)]
    total = counts.sum()
    labels = [str(lbl) for lbl in counts.index]
    values = list(counts.values)

    if horizontal:
        bars = ax.barh(labels, values, color=bar_color, edgecolor="white")
        ax.set_xlabel("Count")
        if xlabel:
            ax.set_ylabel(xlabel)
        if pct:
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_width() + total * 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val / total:.1%}",
                    va="center",
                    fontsize=9,
                )
    else:
        bars = ax.bar(labels, values, color=bar_color, edgecolor="white")
        ax.set_ylabel("Count")
        if xlabel:
            ax.set_xlabel(xlabel)
        if pct:
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.003,
                    f"{val / total:.1%}",
                    ha="center",
                    fontsize=9,
                )

    ax.set_title(title or "")
    return ax
