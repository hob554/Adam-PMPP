"""
多随机种子稳定性分析 - 批量训练脚本
适配 main_lightning.py

运行方式:
    python run_multiseed.py                             # 跑全部
    python run_multiseed.py --dataset cifar10           # 只跑CIFAR-10
    python run_multiseed.py --dataset plantvillage      # 只跑PlantVillage
    python run_multiseed.py --optimizers adam adampmpp  # 只跑指定优化器
    python run_multiseed.py --dry-run                   # 只打印计划不执行

特性:
    - CIFAR-10 数据集自动下载并整理（无需手动操作）
    - 断点续跑: 已存在的CSV自动跳过
    - 实时显示进度和预估剩余时间
    - 日志自动归档到 logs/multiseed/
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ============================================================
# 实验配置
# ============================================================

SEEDS = [1234, 5678, 9012, 4321, 8765]

OPTIMIZER_CONFIGS = {
    "adampmpp":  dict(lr=0.00015, batch_size=128),
    "adam":      dict(lr=0.001,   batch_size=256),
    "sgd":       dict(lr=0.01,    batch_size=256),
    "radam":     dict(lr=0.001,   batch_size=256),
    "adabelief": dict(lr=0.001,   batch_size=256),
}

DATASET_CONFIGS = {
    "plantvillage": dict(
        data_dir="data/PlantVillage",
        max_epochs=50,
        num_workers=4,
    ),
    "cifar10": dict(
        data_dir="data/CIFAR10",
        max_epochs=100,
        num_workers=4,
    ),
}

RESULTS_BASE = Path("results/multiseed")
LOGS_BASE    = Path("logs/multiseed")


# ============================================================
# CIFAR-10 自动下载
# ============================================================

def download_cifar10(data_dir: str):
    """
    自动下载 CIFAR-10 并整理成 train/val 文件夹结构
    结构:
        data_dir/train/{class_name}/{idx}.png
        data_dir/val/{class_name}/{idx}.png
    """
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir   = data_path / "val"

    if train_dir.exists() and val_dir.exists():
        print(f"✅ CIFAR-10 已存在: {data_path}")
        return

    print(f"📥 CIFAR-10 未找到，开始自动下载并整理...")
    print(f"   目标路径: {data_path}")

    try:
        from torchvision import datasets
    except ImportError:
        print("❌ 需要 torchvision，请先安装: pip install torchvision")
        sys.exit(1)

    try:
        from PIL import Image
    except ImportError:
        print("❌ 需要 Pillow，请先安装: pip install Pillow")
        sys.exit(1)

    data_path.mkdir(parents=True, exist_ok=True)
    raw_dir = data_path / "_raw"

    classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    print("   正在下载 CIFAR-10（约170MB，请保持网络连接）...")
    train_dataset = datasets.CIFAR10(root=str(raw_dir), train=True,  download=True)
    val_dataset   = datasets.CIFAR10(root=str(raw_dir), train=False, download=True)

    for split_name, dataset in [("train", train_dataset), ("val", val_dataset)]:
        split_dir = data_path / split_name
        print(f"   整理 {split_name} 集（{len(dataset)} 张图片）...")

        for cls in classes:
            (split_dir / cls).mkdir(parents=True, exist_ok=True)

        for idx, (img, label) in enumerate(dataset):
            cls_name = classes[label]
            img_path = split_dir / cls_name / f"{idx:05d}.png"
            if not img_path.exists():
                img.save(img_path)
            if (idx + 1) % 5000 == 0:
                print(f"   进度: {idx+1}/{len(dataset)}")

        print(f"   ✅ {split_name} 集整理完成")

    print(f"✅ CIFAR-10 下载并整理完成: {data_path}\n")


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


def build_command(optimizer, seed, opt_cfg, ds_cfg, csv_path) -> list:
    return [
        sys.executable, "src/main_lightning.py",
        "--optimizer",   optimizer,
        "--lr",          str(opt_cfg["lr"]),
        "--batch_size",  str(opt_cfg["batch_size"]),
        "--seed",        str(seed),
        "--max_epochs",  str(ds_cfg["max_epochs"]),
        "--num_workers", str(ds_cfg["num_workers"]),
        "--data_dir",    ds_cfg["data_dir"],
        "--output_csv",  str(csv_path),
        "--no_early_stop",
    ]


def run_one(dataset, optimizer, seed, opt_cfg, ds_cfg, dry_run) -> tuple:
    csv_dir = RESULTS_BASE / dataset / "csv"
    log_dir = LOGS_BASE / dataset
    csv_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_dir / f"{optimizer}_seed{seed}.csv"
    log_path = log_dir / f"{optimizer}_seed{seed}.log"

    if csv_path.exists():
        print(f"  ⏭  [{dataset} | {optimizer} | seed={seed}] 已存在，跳过")
        return False, 0.0

    cmd = build_command(optimizer, seed, opt_cfg, ds_cfg, csv_path)
    print(f"  ▶  [{dataset} | {optimizer} | seed={seed}]")
    print(f"     输出: {csv_path}")

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
        print(f"  ❌ 失败 (returncode={result.returncode})，日志: {log_path}")
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
    parser.add_argument("--dataset", nargs="+",
                        default=["plantvillage", "cifar10"],
                        choices=["plantvillage", "cifar10"])
    parser.add_argument("--optimizers", nargs="+",
                        default=list(OPTIMIZER_CONFIGS.keys()),
                        choices=list(OPTIMIZER_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    DATASET_CONFIGS["plantvillage"]["data_dir"] = str(args.data_root / "PlantVillage")
    DATASET_CONFIGS["cifar10"]["data_dir"] = str(args.data_root / "CIFAR10")

    datasets   = args.dataset
    optimizers = args.optimizers
    seeds      = args.seeds
    total      = len(datasets) * len(optimizers) * len(seeds)

    print("=" * 70)
    print(f"多随机种子稳定性分析")
    print(f"数据集: {datasets}")
    print(f"优化器: {optimizers}")
    print(f"种子:   {seeds}")
    print(f"总计:   {total} 次训练")
    print("=" * 70)

    # 自动检查并下载 CIFAR-10
    for dataset in datasets:
        if dataset == "cifar10" and not args.dry_run:
            download_cifar10(DATASET_CONFIGS["cifar10"]["data_dir"])

    # 检查断点续跑
    done = sum(
        (RESULTS_BASE / ds / "csv" / f"{opt}_seed{s}.csv").exists()
        for ds in datasets for opt in optimizers for s in seeds
    )
    if done > 0:
        print(f"📋 检测到 {done}/{total} 次已完成，将自动跳过\n")

    overall_start = time.time()
    new_runs = 0
    accumulated_time = 0.0
    idx = 0

    for dataset in datasets:
        ds_cfg = DATASET_CONFIGS[dataset]
        for optimizer in optimizers:
            opt_cfg = OPTIMIZER_CONFIGS[optimizer]
            for seed in seeds:
                idx += 1
                print(f"\n[{idx}/{total}] ", end="")
                ran, elapsed = run_one(dataset, optimizer, seed,
                                       opt_cfg, ds_cfg, args.dry_run)
                if ran:
                    new_runs += 1
                    accumulated_time += elapsed
                    remaining = total - idx
                    if remaining > 0:
                        avg = accumulated_time / new_runs
                        eta = avg * remaining
                        print(f"     预估剩余: {format_time(eta)}"
                              f"（{remaining}次 × ~{format_time(avg)}）")

    print("\n" + "=" * 70)
    print(f"🎉 完成！新跑了 {new_runs} 次，"
          f"总耗时 {format_time(time.time() - overall_start)}")
    print(f"📊 下一步: python analyze_multiseed.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
