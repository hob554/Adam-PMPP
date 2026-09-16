from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import apply_style, save_figure


OPTIMIZERS = ["adampmpp", "adam", "sgd", "radam", "adabelief"]
OPT_STYLE = {
    "adampmpp": ("Adam-PMPP", "#D55E00", "-", "o"),
    "adam": ("Adam", "#0072B2", "--", "s"),
    "sgd": ("SGD", "#009E73", "-.", "^"),
    "radam": ("RAdam", "#CC79A7", ":", "D"),
    "adabelief": ("AdaBelief", "#222222", "-", "v"),
}

ABLATION = [
    ("baseline", "Adam baseline", "#0072B2", "--"),
    ("full", "Full", "#D55E00", "-"),
    ("wo_powermean", "w/o gradient consistency", "#E69F00", "-."),
    ("wo_warmup", "w/o warmup", "#009E73", ":"),
    ("wo_gradnorm", "w/o norm controller", "#CC79A7", "--"),
    ("only_powermean", "Gradient consistency only", "#56B4E9", "-"),
]

SEEDS = [1234, 5678, 9012, 4321, 8765]
ABLATION_SEEDS = [1234, 5678, 9012]


def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Sanity validation and the first completed epoch share epoch=1.
    # Keep the completed row without dropping a legitimate zero-accuracy epoch.
    return df.drop_duplicates("epoch", keep="last").sort_values("epoch").copy()


def aligned_arrays(paths: list[Path], column: str) -> tuple[np.ndarray, np.ndarray]:
    runs = [load_history(path).set_index("epoch")[column] for path in paths]
    common = runs[0].index
    for run in runs[1:]:
        common = common.intersection(run.index)
    common = common.sort_values()
    values = np.stack([run.loc[common].to_numpy(dtype=float) for run in runs])
    return common.to_numpy(dtype=int), values


def make_single_seed(data_root: Path, dataset: str, output_pdf: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.25))
    csv_dir = data_root / "results" / "multiseed" / dataset / "csv"
    for opt in OPTIMIZERS:
        label, color, linestyle, marker = OPT_STYLE[opt]
        df = load_history(csv_dir / f"{opt}_seed1234.csv")
        ax.plot(
            df["epoch"], df["val_acc"], label=label, color=color,
            linestyle=linestyle, linewidth=1.5, marker=marker,
            markevery=max(1, len(df) // 9), markersize=3.5,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.text(0.01, 0.98, "seed = 1234", transform=ax.transAxes, va="top", fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_pdf)
    plt.close(fig)


def make_multiseed(data_root: Path, dataset: str, output_pdf: Path) -> None:
    apply_style()
    csv_dir = data_root / "results" / "multiseed" / dataset / "csv"
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3))
    metrics = [
        ("train_loss", "Training loss"),
        ("val_loss", "Validation loss"),
        ("val_acc", "Validation accuracy (%)"),
    ]
    cached = {}
    for opt in OPTIMIZERS:
        paths = [csv_dir / f"{opt}_seed{seed}.csv" for seed in SEEDS]
        cached[opt] = {}
        for column, _ in metrics:
            epochs, values = aligned_arrays(paths, column)
            cached[opt][column] = (epochs, values)

    for ax, (column, ylabel), panel in zip(axes.flat[:3], metrics, ["(a)", "(b)", "(c)"]):
        for opt in OPTIMIZERS:
            label, color, linestyle, marker = OPT_STYLE[opt]
            epochs, values = cached[opt][column]
            if column == "train_loss":
                # This callback records the previous training epoch's aggregate.
                # The final training epoch's loss is not present in the CSV.
                epochs = epochs - 1
            finite_columns = np.isfinite(values).any(axis=0)
            epochs = epochs[finite_columns]
            values = values[:, finite_columns]
            mean = np.nanmean(values, axis=0)
            std = np.nanstd(values, axis=0, ddof=0)
            valid = np.isfinite(mean)
            ax.plot(epochs[valid], mean[valid], label=label, color=color,
                    linestyle=linestyle, linewidth=1.35, marker=marker,
                    markevery=max(1, valid.sum() // 8), markersize=3)
            ax.fill_between(epochs[valid], mean[valid] - std[valid], mean[valid] + std[valid],
                            color=color, alpha=0.12, linewidth=0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.text(0.01, 0.98, panel, transform=ax.transAxes, va="top", fontweight="bold")

    axes[0, 0].legend(frameon=False, ncol=2, loc="upper right")
    ax = axes[1, 1]
    means, errors, labels, colors = [], [], [], []
    for opt in OPTIMIZERS:
        label, color, _, _ = OPT_STYLE[opt]
        _, values = cached[opt]["val_acc"]
        means.append(float(np.mean(np.mean(values, axis=0)[-5:])))
        errors.append(float(np.mean(np.std(values, axis=0, ddof=0)[-5:])))
        labels.append(label)
        colors.append(color)
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=errors, capsize=3, color=colors,
                  edgecolor="black", linewidth=0.6, alpha=0.9)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Final-five-epoch accuracy (%)")
    ax.text(0.01, 0.98, "(d)", transform=ax.transAxes, va="top", fontweight="bold")
    ymin = min(means) - 1.0
    ymax = max(m + e for m, e in zip(means, errors)) + 0.8
    ax.set_ylim(ymin, ymax)
    for bar, value, error in zip(bars, means, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + error + 0.08,
                f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    save_figure(fig, output_pdf)
    plt.close(fig)


def make_ablation(data_root: Path, output_pdf: Path) -> None:
    apply_style()
    csv_dir = data_root / "src" / "results" / "ablation" / "csv"
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3))
    cache = {}
    for key, label, color, linestyle in ABLATION:
        paths = [csv_dir / f"{key}_seed{seed}.csv" for seed in ABLATION_SEEDS]
        cache[key] = {
            "val_acc": aligned_arrays(paths, "val_acc"),
            "val_loss": aligned_arrays(paths, "val_loss"),
        }

    for ax, column, ylabel, panel, lower_epoch in [
        (axes[0, 0], "val_acc", "Validation accuracy (%)", "(a)", None),
        (axes[0, 1], "val_loss", "Validation loss", "(b)", None),
        (axes[1, 0], "val_acc", "Validation accuracy (%)", "(c)", 30),
    ]:
        for key, label, color, linestyle in ABLATION:
            epochs, values = cache[key][column]
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0, ddof=1)
            keep = epochs >= lower_epoch if lower_epoch is not None else np.ones_like(epochs, dtype=bool)
            ax.plot(epochs[keep], mean[keep], label=label, color=color,
                    linestyle=linestyle, linewidth=1.25)
            ax.fill_between(epochs[keep], mean[keep] - std[keep], mean[keep] + std[keep],
                            color=color, alpha=0.10, linewidth=0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.text(0.01, 0.98, panel, transform=ax.transAxes, va="top", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=6.5, ncol=2, loc="lower right")

    ax = axes[1, 1]
    means, errors, labels, colors = [], [], [], []
    for key, label, color, _ in ABLATION:
        _, values = cache[key]["val_acc"]
        per_seed_best = values.max(axis=1)
        means.append(float(per_seed_best.mean()))
        errors.append(float(per_seed_best.std(ddof=1)))
        labels.append(label)
        colors.append(color)
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=errors, capsize=3, color=colors,
                  edgecolor="black", linewidth=0.6, alpha=0.9)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_ylim(min(means) - 0.5, max(m + e for m, e in zip(means, errors)) + 0.35)
    ax.text(0.01, 0.98, "(d)", transform=ax.transAxes, va="top", fontweight="bold")
    for bar, value, error in zip(bars, means, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + error + 0.03,
                f"{value:.2f}", ha="center", va="bottom", fontsize=6.5)
    fig.tight_layout()
    save_figure(fig, output_pdf)
    plt.close(fig)
