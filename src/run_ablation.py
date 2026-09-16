#!/usr/bin/env python
"""
消融实验循环执行脚本 (按论文原稿配置 v2)

跑 6 变体 × 3 seeds = 18 次训练:
  1. full           : Adam-PMPP 全开 (PowerMean + Warmup + GradNorm)        [PMPP 系]
  2. wo_powermean   : 关 PowerMean, 留 Warmup + GradNorm                    [PMPP 系]
  3. wo_warmup      : 关 Warmup, 留 PowerMean + GradNorm                    [PMPP 系]
  4. wo_gradnorm    : 关 GradNorm, 留 PowerMean + Warmup                    [PMPP 系]
  5. only_powermean : 只开 PowerMean                                         [PMPP 系]
  6. baseline       : 标准 Adam, 独立超参 (lr=1e-3, batch=256)              [Baseline]

⚠️ 关键: baseline 用的是标准 Adam + 独立调过的超参, 与 PMPP 系不同.
论文原稿: "Adam baseline uses its independently tuned learning rate of 1.0×10⁻³ ...
batch size ... 256 for the Adam baseline".

特性:
  - 断点续跑: 已存在的 csv 自动跳过
  - 日志归档: 每次训练的 stdout 保存到 logs/ablation/{cond}_seed{N}.log
  - 时间统计: 实时打印剩余预估时间

运行:
    python run_ablation.py
    python run_ablation.py --dry-run                 # 只打印计划, 不执行
    python run_ablation.py --only full wo_warmup     # 只跑某些条件
    python run_ablation.py --seeds 42 123            # 自定义 seed
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ============================================================
# 实验配置
# ============================================================

# PMPP 系 5 个变体的开关组合
# 每个 dict 三个布尔值对应 (powermean, warmup, gradnorm)
PMPP_VARIANTS = {
    "full":           dict(powermean=True,  warmup=True,  gradnorm=True),
    "wo_powermean":   dict(powermean=False, warmup=True,  gradnorm=True),
    "wo_warmup":      dict(powermean=True,  warmup=False, gradnorm=True),
    "wo_gradnorm":    dict(powermean=True,  warmup=True,  gradnorm=False),
    "only_powermean": dict(powermean=True,  warmup=False, gradnorm=False),
}

# baseline 单独配置 (与 PMPP 系不共享超参!)
BASELINE_NAME = "baseline"

# 默认种子 (与论文一致)
DEFAULT_SEEDS = [1234, 5678, 9012]

# PMPP 系的训练超参 (论文 Table tab:ablation-variants)
PMPP_HYPERPARAMS = dict(
    optimizer="adampmpp",
    lr=0.00015,         # 1.5e-4
    batch_size=128,
    max_epochs=50,
    num_workers=4,
    data_dir="../data/PlantVillage",
)

# baseline 的训练超参 (独立调参, 论文明确要求)
BASELINE_HYPERPARAMS = dict(
    optimizer="adam",   # 标准 Adam, 不走 adampmpp 路径
    lr=0.001,           # 1.0e-3
    batch_size=256,
    max_epochs=50,
    num_workers=4,
    data_dir="../data/PlantVillage",
)

# 输出路径
RESULTS_DIR = Path("results/ablation/csv")
LOGS_DIR    = Path("logs/ablation")


# ============================================================
# 命令构建
# ============================================================

def build_pmpp_command(variant: str, seed: int, switches: dict, output_csv: Path) -> list:
    """构建 PMPP 系一次训练的命令行"""
    cmd = [
        sys.executable, "main_lightning.py",
        "--optimizer",   PMPP_HYPERPARAMS["optimizer"],
        "--lr",          str(PMPP_HYPERPARAMS["lr"]),
        "--batch_size",  str(PMPP_HYPERPARAMS["batch_size"]),
        "--max_epochs",  str(PMPP_HYPERPARAMS["max_epochs"]),
        "--num_workers", str(PMPP_HYPERPARAMS["num_workers"]),
        "--data_dir",    PMPP_HYPERPARAMS["data_dir"],
        "--seed",        str(seed),
        "--output_csv",  str(output_csv),
        "--no_early_stop",   # 跑满 50 轮, 保证条件间可比
    ]
    cmd.append("--use_powermean" if switches["powermean"] else "--no_powermean")
    cmd.append("--use_warmup"    if switches["warmup"]    else "--no_warmup")
    cmd.append("--use_gradnorm"  if switches["gradnorm"]  else "--no_gradnorm")
    return cmd


def build_baseline_command(seed: int, output_csv: Path) -> list:
    """构建 baseline (标准 Adam) 一次训练的命令行"""
    cmd = [
        sys.executable, "main_lightning.py",
        "--optimizer",   BASELINE_HYPERPARAMS["optimizer"],   # 'adam'
        "--lr",          str(BASELINE_HYPERPARAMS["lr"]),     # 1e-3
        "--batch_size",  str(BASELINE_HYPERPARAMS["batch_size"]),  # 256
        "--max_epochs",  str(BASELINE_HYPERPARAMS["max_epochs"]),
        "--num_workers", str(BASELINE_HYPERPARAMS["num_workers"]),
        "--data_dir",    BASELINE_HYPERPARAMS["data_dir"],
        "--seed",        str(seed),
        "--output_csv",  str(output_csv),
        "--no_early_stop",
        # 三开关随便给(对 'adam' 优化器路径不生效), 给个明确的全关避免歧义
        "--no_powermean", "--no_warmup", "--no_gradnorm",
    ]
    return cmd


# ============================================================
# 工具函数
# ============================================================

def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def run_one(variant: str, seed: int, cmd: list, dry_run: bool) -> tuple[bool, float]:
    """
    跑一次训练.
    返回 (是否新跑了训练, 耗时秒数). 如果是断点跳过, 返回 (False, 0).
    """
    csv_path = RESULTS_DIR / f"{variant}_seed{seed}.csv"
    log_path = LOGS_DIR    / f"{variant}_seed{seed}.log"

    # 断点续跑: csv 已存在就跳过
    if csv_path.exists():
        print(f"  ⏭  [{variant} | seed={seed}] 已存在, 跳过 ({csv_path})")
        return False, 0.0

    print(f"  ▶  [{variant} | seed={seed}]")
    print(f"     输出: {csv_path}")
    print(f"     日志: {log_path}")
    if dry_run:
        print(f"     命令: {' '.join(cmd)}")
        return False, 0.0

    start = time.time()
    with open(log_path, "w") as logf:
        logf.write(f"# Command: {' '.join(cmd)}\n\n")
        logf.flush()
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  ❌ 训练失败 (returncode={result.returncode})")
        print(f"     最后 30 行日志见: {log_path}")
        try:
            tail = subprocess.run(
                ["tail", "-30", str(log_path)],
                capture_output=True, text=True
            ).stdout
            print(tail)
        except Exception:
            pass
        # 删掉空 csv 避免下次误判已完成
        if csv_path.exists() and csv_path.stat().st_size < 100:
            csv_path.unlink()
        return False, elapsed

    print(f"  ✅ 完成 (耗时 {format_time(elapsed)})")
    return True, elapsed


# ============================================================
# 主逻辑
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划, 不实际执行")
    parser.add_argument("--only", nargs="+", default=None,
                        help=f"只跑指定变体 (可选: {list(PMPP_VARIANTS) + [BASELINE_NAME]})")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help=f"自定义 seed (默认 {DEFAULT_SEEDS})")
    parser.add_argument("--data-dir", default="../data/PlantVillage")
    args = parser.parse_args()
    PMPP_HYPERPARAMS["data_dir"] = args.data_dir
    BASELINE_HYPERPARAMS["data_dir"] = args.data_dir

    # 全部变体名(用于校验 --only)
    all_variants = list(PMPP_VARIANTS.keys()) + [BASELINE_NAME]

    # 选择要跑的变体
    if args.only:
        unknown = [c for c in args.only if c not in all_variants]
        if unknown:
            sys.exit(f"❌ 未知变体: {unknown}. 可选: {all_variants}")
        selected = args.only
    else:
        selected = all_variants

    seeds = args.seeds
    total = len(selected) * len(seeds)

    # 创建输出目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # 打印总计划
    print("=" * 70)
    print(f"消融实验 (按论文原稿配置): {len(selected)} 变体 × {len(seeds)} seeds = {total} 次训练")
    print("=" * 70)
    print(f"变体: {selected}")
    print(f"种子: {seeds}")
    print(f"PMPP 系超参: lr={PMPP_HYPERPARAMS['lr']}, bs={PMPP_HYPERPARAMS['batch_size']}, "
          f"epochs={PMPP_HYPERPARAMS['max_epochs']}, no_early_stop")
    print(f"Baseline 超参: optim=adam, lr={BASELINE_HYPERPARAMS['lr']}, "
          f"bs={BASELINE_HYPERPARAMS['batch_size']} (论文公平比较要求)")
    print(f"数据: {PMPP_HYPERPARAMS['data_dir']}")
    print(f"结果: {RESULTS_DIR}/")
    print(f"日志: {LOGS_DIR}/")
    print("=" * 70)

    # 检查已完成数量
    done_count = sum(
        (RESULTS_DIR / f"{v}_seed{s}.csv").exists()
        for v in selected for s in seeds
    )
    if done_count > 0:
        print(f"📋 检测到 {done_count}/{total} 次训练已完成, 将跳过这些\n")

    # 主循环
    overall_start = time.time()
    new_runs = 0
    accumulated_time = 0.0

    idx = 0
    for variant in selected:
        for seed in seeds:
            idx += 1
            print(f"\n[{idx}/{total}] ", end="")

            # 构建命令
            if variant == BASELINE_NAME:
                csv_path = RESULTS_DIR / f"{variant}_seed{seed}.csv"
                cmd = build_baseline_command(seed, csv_path)
            else:
                switches = PMPP_VARIANTS[variant]
                csv_path = RESULTS_DIR / f"{variant}_seed{seed}.csv"
                cmd = build_pmpp_command(variant, seed, switches, csv_path)

            ran, elapsed = run_one(variant, seed, cmd, args.dry_run)

            if ran:
                new_runs += 1
                accumulated_time += elapsed
                remaining_runs = total - idx
                if new_runs > 0 and remaining_runs > 0:
                    avg = accumulated_time / new_runs
                    eta = avg * remaining_runs
                    print(f"     预估剩余: {format_time(eta)} "
                          f"({remaining_runs} 次 × ~{format_time(avg)})")

    # 收尾
    print("\n" + "=" * 70)
    print(f"🎉 全部完成. 实际新跑了 {new_runs} 次训练, "
          f"总耗时 {format_time(time.time() - overall_start)}")
    print(f"📊 下一步: python analyze_ablation.py")
    print("=" * 70)


if __name__ == "__main__":
    main()