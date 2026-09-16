import warnings

warnings.filterwarnings('ignore')

import logging

logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)

import os
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path


class PlantDiseaseDataset(Dataset):
    """植物病害数据集"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(data_dir)
                               if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((img_path, self.class_to_idx[cls]))

        print(f"Loaded {len(self.samples)} samples for {os.path.basename(data_dir)}")
        print(f"Found {len(self.classes)} classes: {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


class MetricsCallback(Callback):
    """自定义回调：收集每个epoch的指标用于保存CSV"""

    def __init__(self):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'time_per_epoch': [],
            'learning_rate': []
        }
        self.epoch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        import time
        self.epoch_start_time = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        import time

        # 获取当前epoch的指标
        metrics = trainer.callback_metrics

        # 记录epoch
        self.history['epoch'].append(trainer.current_epoch + 1)

        # 记录训练loss和acc
        train_loss = metrics.get('train_loss', None)
        self.history['train_loss'].append(train_loss.item() if train_loss is not None else np.nan)

        # Lightning没有自动记录train_acc，我们从模型中获取
        train_acc = getattr(pl_module, 'last_train_acc', np.nan)
        self.history['train_acc'].append(train_acc)

        # 记录验证loss和acc
        val_loss = metrics.get('val_loss', None)
        val_acc = metrics.get('val_acc', None)
        self.history['val_loss'].append(val_loss.item() if val_loss is not None else np.nan)
        self.history['val_acc'].append(val_acc.item() * 100 if val_acc is not None else np.nan)  # 转换为百分比

        # 记录时间
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        self.history['time_per_epoch'].append(epoch_time)

        # 记录学习率
        lr = trainer.optimizers[0].param_groups[0]['lr']
        self.history['learning_rate'].append(lr)


class PlantDiseaseModel(pl.LightningModule):
    """植物病害检测模型 - 终极版:冻结层 + 多层Dropout"""

    def __init__(self, num_classes=15, lr=0.001, optimizer='adam', freeze_backbone=True,
                 use_powermean=True, use_warmup=True, use_gradnorm=True, **kwargs):
        super().__init__()
        self.lr = lr
        self.optimizer_name = optimizer
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        # 消融实验开关 (仅对 optimizer='adampmpp' 生效)
        self.use_powermean = use_powermean
        self.use_warmup = use_warmup
        self.use_gradnorm = use_gradnorm

        # ResNet18 模型
        from torchvision.models import resnet18
        self.model = resnet18(pretrained=True)

        # 🔥 关键改进1:冻结前3个stage,只训练layer4和FC层
        if self.freeze_backbone:
            print("🔒 Freezing layers: conv1, bn1, layer1, layer2, layer3")
            for name, param in self.model.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False

            # 统计可训练参数
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"✅ Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

        # 🔥 关键改进2:多层Dropout + 中间隐藏层
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.7),  # 第一层:70% Dropout
            torch.nn.Linear(num_ftrs, 256),  # 中间层
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),  # 添加BatchNorm稳定训练
            torch.nn.Dropout(0.5),  # 第二层:50% Dropout
            torch.nn.Linear(256, num_classes)
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

        # 导入 Accuracy
        try:
            from torchmetrics import Accuracy
            self.acc = Accuracy(task='multiclass', num_classes=num_classes)
        except:
            self.acc = None

        self.train_losses = []
        self.val_losses = []
        self.val_accs = []

        # 用于记录训练准确率
        self.last_train_acc = 0.0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_losses.append(loss.item())

        # 计算训练准确率
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.last_train_acc = acc.item() * 100  # 转换为百分比

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.val_losses.append(loss.item())
        self.val_accs.append(acc.item())

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """每个验证 epoch 结束打印结果"""
        if self.train_losses and self.val_losses:
            avg_train_loss = np.mean(self.train_losses[-100:])
            avg_val_loss = np.mean(self.val_losses[-10:])
            avg_val_acc = np.mean(self.val_accs[-10:])
            print(
                f"Epoch {self.current_epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_acc={avg_val_acc:.4f}")

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""

        # 🔥 关键改进3:更强的weight_decay
        weight_decay = 5e-4  # 从1e-4增加到5e-4

        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        elif self.optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        elif self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9,
                                        weight_decay=weight_decay, nesterov=True)
        elif self.optimizer_name == 'adampmpp':
            from adam_pmpp import AdamPMPP
            optimizer = AdamPMPP(
                self.parameters(), lr=self.lr, weight_decay=weight_decay,
                use_powermean=self.use_powermean,
                use_warmup=self.use_warmup,
                use_gradnorm=self.use_gradnorm,
            )
        elif self.optimizer_name == 'adabelief':
            try:
                from adabelief_pytorch import AdaBelief
                optimizer = AdaBelief(self.parameters(), lr=self.lr, weight_decay=weight_decay,
                                      weight_decouple=True, rectify=False)
            except ImportError:
                print("⚠️ AdaBelief not found, falling back to Adam")
                optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        elif self.optimizer_name == 'radam':
            try:
                optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
            except AttributeError:
                print("⚠️ RAdam not available in this PyTorch version, falling back to Adam")
                optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        elif self.optimizer_name == 'nadam':
            try:
                optimizer = torch.optim.NAdam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
            except AttributeError:
                print("⚠️ NAdam not available in this PyTorch version, falling back to Adam")
                optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)

        # 学习率调度器 (修复: 移除 verbose 参数以兼容新版 PyTorch)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_acc',
                'interval': 'epoch',
                'frequency': 1
            }
        }


class PlantDiseaseDataModule(pl.LightningDataModule):
    """数据模块 - 增强版"""

    def __init__(self, data_dir='./data/PlantVillage', batch_size=32, num_workers=0, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 强力数据增强
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.15))
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')

        self.train_dataset = PlantDiseaseDataset(train_dir, self.train_transform)
        self.val_dataset = PlantDiseaseDataset(val_dir, self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None
        )


def main(args):
    """主函数"""
    pl.seed_everything(args.seed)

    # 创建数据模块
    data_module = PlantDiseaseDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 创建模型
    model = PlantDiseaseModel(
        num_classes=15,
        lr=args.lr,
        optimizer=args.optimizer,
        freeze_backbone=args.freeze_backbone,
        use_powermean=args.use_powermean,
        use_warmup=args.use_warmup,
        use_gradnorm=args.use_gradnorm,
    )

    # 🔥 添加指标收集回调
    metrics_callback = MetricsCallback()

    # EarlyStopping 改为可选 (消融实验需要关掉以保证所有条件训练长度一致)
    callbacks = [metrics_callback]
    early_stop = None
    if not args.no_early_stop:
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=8,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop)

    # 创建 Trainer
    try:
        trainer = Trainer(
            accelerator='gpu',
            devices=1,
            precision='16-mixed',
            max_epochs=args.max_epochs,
            gradient_clip_val=1.0,
            callbacks=callbacks,
            enable_progress_bar=True,  # ← 改成 True 显示进度
            logger=TensorBoardLogger('lightning_logs', name='plant_disease')
        )
    except:
        print("⚠️ Using legacy precision format (precision=16)")
        trainer = Trainer(
            accelerator='gpu',
            devices=1,
            precision=16,
            max_epochs=args.max_epochs,
            gradient_clip_val=1.0,
            callbacks=callbacks,
            enable_progress_bar=True,  # ← 改成 True 显示进度
            logger=TensorBoardLogger('lightning_logs', name='plant_disease')
        )

    # 训练
    trainer.fit(model, data_module)

    print("\n" + "=" * 60)
    print("🎉 Training completed!")
    if early_stop is not None:
        print(f"Best model stopped at epoch: {early_stop.stopped_epoch if early_stop.stopped_epoch > 0 else 'N/A'}")
    else:
        print(f"Trained full {args.max_epochs} epochs (early stopping disabled)")
    print("=" * 60)

    # 🔥 保存训练历史到CSV（如果指定了输出路径）
    if hasattr(args, 'output_csv') and args.output_csv:
        df = pd.DataFrame(metrics_callback.history)

        # 确保目录存在
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存CSV
        df.to_csv(output_path, index=False)
        print(f"✅ Training history saved to: {output_path}")

        # 打印最终结果
        if len(df) > 0:
            best_val_acc = df['val_acc'].max()
            best_epoch = df.loc[df['val_acc'].idxmax(), 'epoch']
            final_val_acc = df['val_acc'].iloc[-1]

            print(f"\n📊 Results Summary:")
            print(f"   Best Val Acc: {best_val_acc:.2f}% (Epoch {int(best_epoch)})")
            print(f"   Final Val Acc: {final_val_acc:.2f}%")
            print(f"   Total Epochs: {len(df)}")


if __name__ == '__main__':
    parser = ArgumentParser()

    # 模型参数
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--optimizer', default='adam', type=str,
                        choices=['adam', 'adamw', 'sgd', 'adampmpp', 'adabelief', 'radam', 'nadam'])
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--max_epochs', default=50, type=int)
    parser.add_argument('--freeze_backbone', default=True, type=bool,
                        help='Freeze ResNet backbone layers (recommended)')

    # 数据参数
    parser.add_argument('--data_dir', default='./data/PlantVillage', type=str)

    # 🔥 新增：输出CSV路径
    parser.add_argument('--output_csv', default=None, type=str,
                        help='Path to save training history CSV')

    # 🔥 消融实验开关 (仅对 --optimizer adampmpp 生效)
    parser.add_argument('--use_powermean', dest='use_powermean', action='store_true',
                        help='Enable PowerMean gradient consistency adjustment')
    parser.add_argument('--no_powermean', dest='use_powermean', action='store_false',
                        help='Disable PowerMean')
    parser.set_defaults(use_powermean=True)

    parser.add_argument('--use_warmup', dest='use_warmup', action='store_true',
                        help='Enable linear warmup')
    parser.add_argument('--no_warmup', dest='use_warmup', action='store_false',
                        help='Disable warmup')
    parser.set_defaults(use_warmup=True)

    parser.add_argument('--use_gradnorm', dest='use_gradnorm', action='store_true',
                        help='Enable gradient-norm-based adaptive lr')
    parser.add_argument('--no_gradnorm', dest='use_gradnorm', action='store_false',
                        help='Disable gradnorm')
    parser.set_defaults(use_gradnorm=True)

    # 🔥 关闭 EarlyStopping (消融实验需要)
    parser.add_argument('--no_early_stop', action='store_true',
                        help='Disable EarlyStopping (for ablation: ensure equal training length)')

    args = parser.parse_args()

    main(args)