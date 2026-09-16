# Adam-PMPP

Code and archived experiment histories for **Adam-PMPP: Gradient-Consistency and Norm-Ratio Modulation in Visual Classification**.

Adam-PMPP augments Adam with per-tensor gradient-direction consistency, linear warmup, and a post-warmup gradient-norm controller. The consistency statistic is the arithmetic mean of adjacent gradient cosine similarities.

## Contents

- `src/adam_pmpp.py`: optimizer implementation.
- `src/main_lightning.py`: shared ResNet-18 training implementation.
- `src/run_ablation.py`: six configurations over three seeds.
- `multiseed/run_multiseed.py`: five optimizers over five seeds on two datasets.
- `results/multiseed/`: 50 archived CSV histories.
- `src/results/ablation/csv/`: 18 archived CSV histories.
- `figures/`: plotting code used by the revised English manuscript.
- `requirements-recorded.txt`: recorded package versions, including the CUDA-specific PyTorch build.

Dataset images, checkpoints, manuscript drafts, and local environment files are not included.

## Recreate the paper figures from archived CSVs

From the repository root, in an environment with NumPy, pandas, and matplotlib:

```bash
python -m pip install numpy pandas matplotlib
python figures/reproduce_figures.py
```

This writes the five experimental result figures to `figures/generated/`. It does not train a model or download data.

To regenerate the ablation table:

```bash
cd src
python analyze_ablation.py
cd ..
```

The ablation statistics are the mean and sample standard deviation of each seed's best validation accuracy. Multiseed summaries average epoch-wise cross-seed means over the final five epochs; their dispersion averages epoch-wise population standard deviations over the same window. These error bars are not confidence intervals.

## Training environment and datasets

The recorded runs used Python 3.13.5, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, PyTorch Lightning 2.6.1, and an NVIDIA RTX 5070 Ti. See `requirements-recorded.txt` for the remaining recorded versions. Training also requires TensorBoard for Lightning's logger. Install a compatible PyTorch/torchvision build for your GPU before installing the remaining dependencies. The CUDA-specific requirements are an environment record, not a universal lock file.

The loader expects:

```text
data/
  PlantVillage/
    train/<class>/*
    val/<class>/*
  CIFAR10/
    train/<class>/*
    val/<class>/*
```

The archived PlantVillage subset contains 15 classes, 16,516 training images and 4,122 validation images. Its original unsplit source and split manifest are not included, so a newly prepared split is not guaranteed to reproduce those exact examples. For CIFAR-10, the runner can download and export the standard train/test partitions into the expected train/val layout.

Preview commands without downloading data or starting training:

```bash
python -X utf8 multiseed/run_multiseed.py --data-root data --dry-run
cd src
python -X utf8 run_ablation.py --data-dir ../data/PlantVillage --dry-run
cd ..
```

Remove `--dry-run` to train. The runners skip CSV files that already exist. To perform fresh runs, use a separate working copy without the archived result CSVs and preserve the originals for comparison. Multiseed commands run from the repository root; ablation commands run from `src/`.

## Experimental interpretation

- The archived callback records the preceding epoch's training loss during validation. The plotting code therefore shifts finite training-loss entries back by one epoch; the final training-epoch loss is unavailable (curves end at epoch 99 for CIFAR-10 and 49 for PlantVillage). Validation metrics and archived CSV files are unchanged. Duplicate epoch rows retain the last record, including genuine zero validation accuracy.
- The archived CIFAR-10 implementation retains a 15-output classifier for its 10 labels and uses the same ImageNet-style preprocessing as PlantVillage.
- Adam-PMPP uses learning rate 0.00015 and batch size 128; Adam, RAdam, and AdaBelief use 0.001 and 256; SGD uses 0.01 and 256. Comparisons do not match update count or wall-clock cost.
- The validation partition is monitored during training and used for reporting. CIFAR-10's standard test partition serves this validation role.
- The no-warmup ablation retains step 101 as the norm controller's activation point.
- AdaBelief has the highest reported final-window mean accuracy in both datasets. Small component-study differences do not establish individual-component necessity.

## Release provenance

The optimizer and training implementation are copied unchanged from the experiment archive. Release preparation only makes runner dataset paths configurable, changes their defaults to relative paths, and prevents the multiseed dry run from downloading CIFAR-10. CSV files are copied unchanged. The plotting source is from the revised English manuscript. `MANIFEST_SHA256.json` records file checksums.

No software license has been selected yet; repository visibility alone does not grant an open-source license.
