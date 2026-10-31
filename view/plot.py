from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from transformers import Trainer

if TYPE_CHECKING:
    from ..evaluate import EvalResult

RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua: colour-blind safe as a set of 3
GRID, INK, SURFACE = "#e6e5e1", "#52514e", "#ffffff"


def _trainer_state(trainer_or_path: Trainer | str | Path) -> dict[str, Any]:
    if isinstance(trainer_or_path, (str, Path)):
        with open(trainer_or_path) as f:
            state: dict[str, Any] = json.load(f)
        return state
    return {
        "log_history": trainer_or_path.state.log_history,
        "best_global_step": trainer_or_path.state.best_global_step,
    }


def run_history(trainer_or_path: Trainer | str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-epoch history from a Trainer or a trainer_state.json path."""
    df = pd.DataFrame(_trainer_state(trainer_or_path)["log_history"])
    evals = df.dropna(subset=["eval_loss"])[["epoch", "step", "eval_loss", "eval_macro_f1"]]
    train = df.dropna(subset=["loss"])[["epoch", "loss"]] if "loss" in df else pd.DataFrame()
    return evals, train


def _selected_epoch(evals: pd.DataFrame, best_step: int | None) -> pd.Series:
    # The checkpoint the Trainer actually kept (follows metric_for_best_model); lowest val loss otherwise.
    if best_step is not None and (evals["step"] == best_step).any():
        return evals[evals["step"] == best_step].iloc[0]
    return evals.iloc[evals["eval_loss"].argmin()]


def plot_run(trainer_or_path: Trainer | str | Path, title: str = "Run") -> None:
    evals, train = run_history(trainer_or_path)
    best = _selected_epoch(evals, _trainer_state(trainer_or_path).get("best_global_step"))
    blue, orange, grid, ink = RUN_COLORS[0], RUN_COLORS[1], GRID, INK

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Loss
    if len(train):
        ax1.plot(train["epoch"], train["loss"], color=orange, lw=2, marker="o", ms=5, label="Training loss")
    ax1.plot(evals["epoch"], evals["eval_loss"], color=blue, lw=2, marker="o", ms=5, label="Validation loss")
    ax1.set_title("Loss")
    ax1.legend(frameon=False)

    # Macro-F1
    ax2.plot(evals["epoch"], evals["eval_macro_f1"], color=blue, lw=2, marker="o", ms=5)
    ax2.set_title("Validation macro-F1")
    ax2.set_ylim(0, 1)

    for ax in (ax1, ax2):
        ax.axvline(best["epoch"], color=ink, lw=1, ls="--")
        ax.set_xlabel("Epoch")
        ax.grid(color=grid, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=ink)
    ax1.text(best["epoch"], ax1.get_ylim()[1], " selected", color=ink, va="top", fontsize=9)

    fig.suptitle(
        f"{title}: selected epoch {int(best['epoch'])}, "
        f"val loss {best['eval_loss']:.3f}, macro-F1 {best['eval_macro_f1']:.3f}"
    )
    fig.tight_layout()
    plt.show()


def compare_runs(runs: dict[str, Trainer | str | Path]) -> None:
    """Validation loss and macro-F1 per epoch for several runs, e.g. {"MuRIL": trainer, "XLM-R": "path"}.

    Each value is a Trainer or a path to its trainer_state.json.
    """
    colors = RUN_COLORS
    if len(runs) > len(colors):
        raise ValueError(f"compare_runs supports up to {len(colors)} runs, got {len(runs)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for (name, src), color in zip(runs.items(), colors, strict=False):
        evals, _ = run_history(src)
        ax1.plot(evals["epoch"], evals["eval_loss"], color=color, lw=2, marker="o", ms=5, label=name)
        ax2.plot(evals["epoch"], evals["eval_macro_f1"], color=color, lw=2, marker="o", ms=5, label=name)
    ax1.set_title("Validation loss")
    ax2.set_title("Validation macro-F1")
    ax2.set_ylim(0, 1)
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.grid(color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def compare_per_class_f1(results: dict[str, EvalResult], title: str = "Per-class F1") -> None:
    """Per-class F1 for up to 3 evaluation results, side by side, e.g. {"MuRIL": r1, "XLM-R": r2}.

    Classes are sorted by the first result's F1; each class shows its test-set support.
    """
    if not 1 <= len(results) <= len(RUN_COLORS):
        raise ValueError(f"compare_per_class_f1 takes 1 to {len(RUN_COLORS)} results, got {len(results)}")

    first = next(iter(results.values()))
    labels = sorted(first.label_names, key=lambda name: first.report[name]["f1-score"])
    for result in results.values():
        labels += [name for name in result.label_names if name not in labels]

    n_runs = len(results)
    bar_height = 0.8 / n_runs
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(labels) * n_runs + 1.5)))
    for i, ((name, result), color) in enumerate(zip(results.items(), RUN_COLORS, strict=False)):
        f1 = [result.report[label]["f1-score"] if label in result.report else np.nan for label in labels]
        offset = ((n_runs - 1) / 2 - i) * bar_height  # first run on top within each class, like the legend
        ax.barh(y + offset, f1, height=bar_height, color=color, edgecolor=SURFACE, linewidth=1, label=name)

    support = {label: int(first.report[label]["support"]) for label in first.label_names}
    ax.set_yticks(y, [f"{label} (n={support[label]})" if label in support else label for label in labels])
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.set_title(title, pad=24 if len(results) > 1 else 6)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)
    if n_runs > 1:
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1), ncol=n_runs, borderaxespad=0.2)
    fig.tight_layout()
    plt.show()


# Single-hue blue ramp: near-white means "none of this class", dark means "most of it".
_CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    "sage_blues", ["#f7f9fc", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def plot_confusion_matrix(result: EvalResult, title: str = "Confusion matrix") -> None:
    """Confusion matrix shaded by share of each true class (row), annotated with raw counts."""
    counts = result.confusion_matrix
    row_totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, row_totals, out=np.zeros(counts.shape, dtype=float), where=row_totals > 0)
    names = result.label_names
    size = max(5.0, 0.55 * len(names) + 2)

    fig, ax = plt.subplots(figsize=(size + 1, size))
    image = ax.imshow(shares, cmap=_CONFUSION_CMAP, vmin=0, vmax=1)
    for row in range(len(names)):
        for col in range(len(names)):
            if counts[row, col]:
                color = SURFACE if shares[row, col] > 0.5 else "#2b2a27"
                ax.text(col, row, str(counts[row, col]), ha="center", va="center", color=color, fontsize=9)

    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(colors=INK, length=0)
    ax.spines[:].set_visible(False)
    accuracy = result.metrics.get("accuracy")
    ax.set_title(f"{title} (accuracy {accuracy:.3f})" if accuracy is not None else title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Share of true class")
    fig.tight_layout()
    plt.show()
