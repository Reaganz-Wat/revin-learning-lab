"""
plotting.py
Reusable Matplotlib/Seaborn helpers consumed by all notebooks.
Notebooks call these functions — no bulky plot code lives in .ipynb cells.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


_DEFAULT_FIGSIZE = (10, 4)
_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


# ══════════════════════════════════════════════════════════════════════════════
# Basic charts
# ══════════════════════════════════════════════════════════════════════════════

def plot_histogram(
    data: np.ndarray,
    title: str = "Histogram",
    xlabel: str = "Value",
    bins: int = 20,
    color: str = _PALETTE[0],
    show_mean: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    else:
        fig = ax.figure

    ax.hist(data, bins=bins, color=color, edgecolor="white", alpha=0.85)
    if show_mean:
        ax.axvline(np.mean(data), color="red", linestyle="--", linewidth=1.5,
                   label=f"mean = {np.mean(data):.2f}")
        ax.legend()
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.spines[["top", "right"]].set_visible(False)
    if standalone:
        plt.tight_layout()
        plt.show()
    return fig


def plot_line(
    x,
    y,
    title: str = "Line Plot",
    xlabel: str = "Index",
    ylabel: str = "Value",
    label: str | None = None,
    color: str = _PALETTE[0],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    else:
        fig = ax.figure

    ax.plot(x, y, color=color, linewidth=1.6, label=label)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    if label:
        ax.legend()
    if standalone:
        plt.tight_layout()
        plt.show()
    return fig


def plot_scatter(
    x,
    y,
    title: str = "Scatter Plot",
    xlabel: str = "X",
    ylabel: str = "Y",
    color: str = _PALETTE[0],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    else:
        fig = ax.figure

    ax.scatter(x, y, color=color, alpha=0.7, edgecolors="white", linewidths=0.5)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    if standalone:
        plt.tight_layout()
        plt.show()
    return fig


def plot_boxplot(
    data_dict: dict[str, np.ndarray],
    title: str = "Box Plot",
    ylabel: str = "Value",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    labels = list(data_dict.keys())
    values = [np.asarray(v) for v in data_dict.values()]
    bp = ax.boxplot(values, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], _PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Time series specific
# ══════════════════════════════════════════════════════════════════════════════

def plot_rolling_stats(
    series: np.ndarray,
    window: int = 20,
    title: str = "Rolling Mean & Std",
    xlabel: str = "Time",
    ylabel: str = "Value",
) -> plt.Figure:
    series = np.asarray(series, dtype=float)
    roll_mean = np.array([
        np.mean(series[max(0, i - window):i + 1]) for i in range(len(series))
    ])
    roll_std = np.array([
        np.std(series[max(0, i - window):i + 1]) for i in range(len(series))
    ])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(series, color=_PALETTE[0], linewidth=1.3, label="Raw")
    axes[0].plot(roll_mean, color=_PALETTE[1], linewidth=2, label=f"Rolling mean (w={window})")
    axes[0].fill_between(
        range(len(series)),
        roll_mean - roll_std,
        roll_mean + roll_std,
        alpha=0.2, color=_PALETTE[1], label="±1 std band",
    )
    axes[0].set_title(title, fontsize=13)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(roll_std, color=_PALETTE[2], linewidth=1.5)
    axes[1].set_title(f"Rolling Std (window={window})", fontsize=11)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Std")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Normalization helpers
# ══════════════════════════════════════════════════════════════════════════════

def plot_normalization_comparison(
    original: np.ndarray,
    zscore: np.ndarray,
    minmax: np.ndarray,
    title: str = "Normalization Comparison",
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    datasets = [
        (original, "Original", _PALETTE[0]),
        (zscore,   "Z-Score",  _PALETTE[1]),
        (minmax,   "Min-Max",  _PALETTE[2]),
    ]
    for ax, (data, label, color) in zip(axes, datasets):
        ax.hist(data, bins=20, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(data), color="black", linestyle="--", linewidth=1.2,
                   label=f"μ={np.mean(data):.2f}")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    return fig


def plot_before_after_distribution(
    before: np.ndarray,
    after: np.ndarray,
    label_before: str = "Before",
    label_after: str = "After",
    title: str = "Distribution Before vs After",
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, data, label, color in zip(
        axes,
        [before, after],
        [label_before, label_after],
        [_PALETTE[0], _PALETTE[1]],
    ):
        ax.hist(data, bins=25, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(data), color="red", linestyle="--", linewidth=1.3,
                   label=f"μ={np.mean(data):.2f}\nσ={np.std(data):.2f}")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
    return fig
