#!/usr/bin/env python
"""
消融实验结果分析脚本

输入: results/ablation/csv/{condition}_seed{seed}.csv  (共 18 个)
输出:
  - results/ablation/summary.csv         逐次明细 (18行)
  - results/ablation/aggregate.csv       按条件汇总 (6行, 含均值±std)
  - results/ablation/figure5.pdf         论文 Figure 5 四联图
  - results/ablation/figure5.png         同上 PNG 版本
  - results/ablation/table_ablation.tex  论文 LaTeX 表格片段

运行: python analyze_ablation.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 配置 (与 run_ablation.py 保持一致)
# ============================================================

CONDITIONS = [
    "full",
    "wo_powermean",
    "wo_warmup",
    "wo_gradnorm",
    "only_powermean",
    "baseline",
]
SEEDS = [1234, 5678, 9012]

# 论文展示名 (用于图例和表头)
PRETTY_NAME = {
    "full":           "Full",
    "wo_powermean":   "w/o gradient consistency",
    "wo_warmup":      "w/o warmup",
    "wo_gradnorm":    "w/o gradient-norm controller",
    "only_powermean": "Gradient consistency only",
    "baseline":       "Adam baseline",
}

# 六条曲线的颜色 (色觉友好 + 黑白可区分)
COLORS = {
    "full":           "#d62728",
    "wo_powermean":   "#1f77b4",
    "wo_warmup":      "#ff7f0e",
    "wo_gradnorm":    "#9467bd",
    "only_powermean": "#2ca02c",
    "baseline":       "#7f7f7f",
}

CSV_DIR    = Path("results/ablation/csv")
OUTPUT_DIR = Path("results/ablation")

# ============================================================
# 1. 加载数据
# ============================================================

def load_all_runs() -> dict:
    """
    返回 {condition: [df_seed1234, df_seed5678, df_seed9012]}
    """
    data = {}
    missing = []
    for cond in CONDITIONS:
        runs = []
        for seed in SEEDS:
            csv = CSV_DIR / f"{cond}_seed{seed}.csv"
            if not csv.exists():
                missing.append(str(csv))
                continue
            df = pd.read_csv(csv)
            # Lightning 在正式训练前写入一行 sanity-validation 记录；该行
            # train_loss 为空且 time_per_epoch 为 0，不属于训练 epoch。
            df = df[df["train_loss"].notna()].copy()
            runs.append(df)
        data[cond] = runs

    if missing:
        print("⚠️ 以下 csv 缺失, 请先运行 run_ablation.py:")
        for m in missing:
            print(f"     {m}")
        sys.exit(1)
    return data


def compute_per_run_metrics(df: pd.DataFrame) -> dict:
    """从单次训练的 epoch-level df 提取关键指标"""
    best_val_acc = df["val_acc"].max()
    best_epoch   = int(df.loc[df["val_acc"].idxmax(), "epoch"])
    final_val_acc = df["val_acc"].iloc[-1]
    final_train_loss = df["train_loss"].iloc[-1]
    avg_time = df["time_per_epoch"].mean()

    return dict(
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        final_val_acc=final_val_acc,
        final_train_loss=final_train_loss,
        avg_time_per_epoch=avg_time,
    )


# ============================================================
# 2. 生成汇总表
# ============================================================

def build_summary(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回:
      summary_df   - 18 行明细 (每条件每seed一行)
      aggregate_df - 6 行汇总 (每条件均值±std)
    """
    rows = []
    for cond, runs in data.items():
        for seed, df in zip(SEEDS, runs):
            metrics = compute_per_run_metrics(df)
            rows.append(dict(condition=cond, seed=seed, **metrics))
    summary_df = pd.DataFrame(rows)

    # 聚合
    agg_rows = []
    for cond in CONDITIONS:
        sub = summary_df[summary_df["condition"] == cond]
        agg_rows.append(dict(
            condition=cond,
            best_val_acc_mean=sub["best_val_acc"].mean(),
            best_val_acc_std =sub["best_val_acc"].std(ddof=1),
            best_epoch_mean=sub["best_epoch"].mean(),
            best_epoch_std =sub["best_epoch"].std(ddof=1),
            final_val_acc_mean=sub["final_val_acc"].mean(),
            final_train_loss_mean=sub["final_train_loss"].mean(),
            avg_time_per_epoch=sub["avg_time_per_epoch"].mean(),
        ))
    aggregate_df = pd.DataFrame(agg_rows)
    return summary_df, aggregate_df


# ============================================================
# 3. 绘制 Figure 5 (四联图)
# ============================================================

def stack_curves(runs: list[pd.DataFrame], col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    把 N 次 run 的曲线对齐到最短长度, 返回 (epochs, mean, std)
    """
    min_len = min(len(df) for df in runs)
    arr = np.stack([df[col].values[:min_len] for df in runs], axis=0)
    epochs = runs[0]["epoch"].values[:min_len]
    return epochs, arr.mean(axis=0), arr.std(axis=0, ddof=1)


def plot_figure5(data: dict, aggregate_df: pd.DataFrame):
    """生成 Figure 5: 4 个子图"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ---- 子图 (a): 训练 loss 曲线 (mean ± std) ----
    ax = axes[0, 0]
    for cond in CONDITIONS:
        epochs, mean, std = stack_curves(data[cond], "train_loss")
        ax.plot(epochs, mean, label=PRETTY_NAME[cond], color=COLORS[cond], linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, color=COLORS[cond], alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("(a) Training Loss Curves")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- 子图 (b): 验证准确率曲线 (mean ± std) ----
    ax = axes[0, 1]
    for cond in CONDITIONS:
        epochs, mean, std = stack_curves(data[cond], "val_acc")
        ax.plot(epochs, mean, label=PRETTY_NAME[cond], color=COLORS[cond], linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, color=COLORS[cond], alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("(b) Validation Accuracy Curves")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- 子图 (c): 每次运行取得最高验证准确率的 epoch ----
    ax = axes[1, 0]
    x_pos = np.arange(len(CONDITIONS))
    means = [aggregate_df.loc[aggregate_df["condition"] == c, "best_epoch_mean"].iloc[0]
             for c in CONDITIONS]
    stds  = [aggregate_df.loc[aggregate_df["condition"] == c, "best_epoch_std"].iloc[0]
             for c in CONDITIONS]
    bars = ax.bar(x_pos, means, yerr=stds, capsize=6,
                  color=[COLORS[c] for c in CONDITIONS],
                  edgecolor="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([PRETTY_NAME[c] for c in CONDITIONS], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Epoch of Best Validation Accuracy")
    ax.set_title("(c) Epoch of Best Accuracy")
    ax.grid(True, alpha=0.3, axis="y")
    # 数值标注
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{m:.1f}", ha="center", fontsize=9)

    # ---- 子图 (d): 模块贡献条形图 (相对 baseline 的提升) ----
    ax = axes[1, 1]
    baseline_acc = aggregate_df.loc[aggregate_df["condition"] == "baseline",
                                    "best_val_acc_mean"].iloc[0]
    deltas = []
    delta_stds = []
    for cond in CONDITIONS:
        row = aggregate_df.loc[aggregate_df["condition"] == cond].iloc[0]
        deltas.append(row["best_val_acc_mean"] - baseline_acc)
        delta_stds.append(row["best_val_acc_std"])
    bars = ax.bar(x_pos, deltas, yerr=delta_stds, capsize=6,
                  color=[COLORS[c] for c in CONDITIONS],
                  edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([PRETTY_NAME[c] for c in CONDITIONS], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Δ Best Val Acc vs Baseline (%)")
    ax.set_title("(d) Module Contribution (higher is better)")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, d in zip(bars, deltas):
        y = bar.get_height()
        offset = 0.05 if y >= 0 else -0.15
        ax.text(bar.get_x() + bar.get_width() / 2, y + offset,
                f"{d:+.2f}", ha="center", fontsize=9)

    fig.suptitle("Ablation Study of Adam-PMPP Components on PlantVillage",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    pdf_path = OUTPUT_DIR / "figure5.pdf"
    png_path = OUTPUT_DIR / "figure5.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  ✅ {pdf_path}")
    print(f"  ✅ {png_path}")


# ============================================================
# 4. 生成 LaTeX 表格
# ============================================================

def write_latex_table(aggregate_df: pd.DataFrame):
    """生成论文用的 LaTeX 三线表"""
    lines = [
        r"% Auto-generated by analyze_ablation.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study of Adam-PMPP components on PlantVillage. "
        r"Results are mean$\pm$std over 3 random seeds. "
        r"Best Val Acc: highest validation accuracy across 50 epochs. "
        r"Best Epoch: epoch at which the highest validation accuracy is first attained.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Condition & Consistency / Warmup / Norm & Best Val Acc (\%) & Best Epoch \\",
        r"\midrule",
    ]
    switch_str = {
        "full":           r"\cmark / \cmark / \cmark",
        "wo_powermean":   r"\xmark / \cmark / \cmark",
        "wo_warmup":      r"\cmark / \xmark / \cmark",
        "wo_gradnorm":    r"\cmark / \cmark / \xmark",
        "only_powermean": r"\cmark / \xmark / \xmark",
        "baseline":       r"\xmark / \xmark / \xmark",
    }
    for cond in CONDITIONS:
        row = aggregate_df.loc[aggregate_df["condition"] == cond].iloc[0]
        acc_mean = row["best_val_acc_mean"]
        acc_std  = row["best_val_acc_std"]
        best_epoch_mean = row["best_epoch_mean"]
        best_epoch_std  = row["best_epoch_std"]
        name = PRETTY_NAME[cond]
        # full 行加粗
        if cond == "full":
            line = (f"\\textbf{{{name}}} & {switch_str[cond]} & "
                    f"$\\mathbf{{{acc_mean:.2f}\\pm{acc_std:.2f}}}$ & "
                    f"$\\mathbf{{{best_epoch_mean:.1f}\\pm{best_epoch_std:.1f}}}$ \\\\")
        else:
            line = (f"{name} & {switch_str[cond]} & "
                    f"${acc_mean:.2f}\\pm{acc_std:.2f}$ & "
                    f"${best_epoch_mean:.1f}\\pm{best_epoch_std:.1f}$ \\\\")
        lines.append(line)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_path = OUTPUT_DIR / "table_ablation.tex"
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"  ✅ {tex_path}")


# ============================================================
# 主入口
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("分析消融实验结果")
    print("=" * 70)

    print("\n[1/4] 加载 csv...")
    data = load_all_runs()
    print(f"  ✅ 加载完成: {sum(len(v) for v in data.values())} 个训练记录")

    print("\n[2/4] 计算汇总指标...")
    summary_df, aggregate_df = build_summary(data)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    aggregate_df.to_csv(OUTPUT_DIR / "aggregate.csv", index=False)
    print(f"  ✅ {OUTPUT_DIR/'summary.csv'} ({len(summary_df)} 行)")
    print(f"  ✅ {OUTPUT_DIR/'aggregate.csv'} ({len(aggregate_df)} 行)")

    # 终端打印关键结果
    print("\n  关键结果 (best_val_acc, mean±std):")
    for cond in CONDITIONS:
        row = aggregate_df.loc[aggregate_df["condition"] == cond].iloc[0]
        marker = " ⭐" if cond == "full" else ""
        print(f"    {PRETTY_NAME[cond]:24s}  "
              f"{row['best_val_acc_mean']:.2f} ± {row['best_val_acc_std']:.2f}   "
              f"(best epoch {row['best_epoch_mean']:.1f}){marker}")

    print("\n[3/4] 绘制 Figure 5...")
    plot_figure5(data, aggregate_df)

    print("\n[4/4] 生成 LaTeX 表格...")
    write_latex_table(aggregate_df)

    print("\n" + "=" * 70)
    print("🎉 分析完成!")
    print(f"📁 全部产物在: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
