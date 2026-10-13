"""遥感图像二值耕地分割训练脚本（GPU强制 + 预训练权重 + 完整指标 + 异常处理）"""
import argparse
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from models.factory import build_model
from utils.metrics import compute_metrics


# ==================== GPU 强制验证 ====================

def assert_gpu():
    """强制要求 CUDA 可用。"""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 不可用。必须在 GPU 环境下运行训练。"
            "请确认: 1) NVIDIA驱动已安装 2) PyTorch CUDA版本正确 3) CUDA_VISIBLE_DEVICES 已设置"
        )
    print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[OK] CUDA version: {torch.version.cuda}")
    print(f"[OK] PyTorch version: {torch.__version__}")


# ==================== 数据集 ====================

class RemoteSensingDataset(Dataset):
    """遥感图像分割数据集（支持按空间划分 train/val）。

    数据目录约定:
        data_dir/
        ├── Shouyang/
        │   ├── train/images/*.npy   # (T, C, H, W)  float32
        │   ├── train/labels/*.npy  # (H, W)       uint8
        │   ├── val/images/*.npy
        │   └── val/labels/*.npy
        └── Zhangzi/
            ├── train/images/*.npy
            ├── train/labels/*.npy
            ├── val/images/*.npy
            └── val/labels/*.npy
    """

    def __init__(self, data_dir, mode='train', num_bands=6, num_temporal=6,
                 region='Shouyang', transform=None):
        self.data_dir = Path(data_dir) / region / mode
        self.images_dir = self.data_dir / 'images'
        self.labels_dir = self.data_dir / 'labels'
        self.num_bands = num_bands
        self.num_temporal = num_temporal
        self.transform = transform

        self.images = sorted(self.images_dir.glob('*.npy'))
        self.labels = sorted(self.labels_dir.glob('*.npy'))

        if len(self.images) == 0:
            raise FileNotFoundError(
                f"未找到数据: {self.images_dir}\n"
                f"请确认数据已按 70:30 空间划分到 train/val 子目录"
            )
        if len(self.images) != len(self.labels):
            raise ValueError(
                f"图像({len(self.images)})与标签({len(self.labels)})数量不匹配"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        # 查找对应的 label 文件
        lbl_path = self.labels_dir / img_path.name.replace('_img', '_lbl')
        if not lbl_path.exists():
            # 尝试同名匹配
            lbl_path = self.labels_dir / img_path.name

        image = np.load(img_path)  # (T, C, H, W)
        label = np.load(lbl_path)  # (H, W)

        # 波段选择: 6->3 取前3波段(RGB: B2, B3, B4)
        if self.num_bands == 3:
            image = image[:, :3, :, :]
        elif self.num_bands != 6:
            raise ValueError(f"不支持的波段数: {self.num_bands}")

        # 时相选择: 6->1 取第一个
        if self.num_temporal == 1:
            image = image[0:1, :, :, :]
        elif self.num_temporal != 6:
            raise ValueError(f"不支持的时相数: {self.num_temporal}")

        image = torch.from_numpy(image.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        if self.transform:
            image, label = self.transform(image, label)

        return image, label


# ==================== 损失函数 ====================

def dice_loss(logits, target, smooth=1.0):
    """Dice Loss (二分类)。"""
    probs = torch.sigmoid(logits[:, 1])  # 耕地类别的概率
    target_f = target.float()
    intersection = (probs * target_f).sum()
    dice = (2. * intersection + smooth) / (probs.sum() + target_f.sum() + smooth)
    return 1 - dice


def ce_dice_loss(logits, target, class_weight=None):
    """加权 CE + Dice。"""
    ce = nn.functional.cross_entropy(logits, target, weight=class_weight)
    dice = dice_loss(logits, target)
    return ce + dice


# ==================== 训练 / 验证 ====================

def train_one_epoch(model, loader, optimizer, device, class_weight=None):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(loader):
        try:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = ce_dice_loss(outputs, labels, class_weight)

            # NaN 检测
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARNING] Batch {batch_idx}: loss is NaN/Inf, skipping. "
                      f"images range=[{images.min():.2f}, {images.max():.2f}]")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.detach().cpu().numpy())
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"  [ERROR] Batch {batch_idx}: CUDA OOM. Skipping batch.")
                torch.cuda.empty_cache()
                continue
            raise

    if num_batches == 0:
        raise RuntimeError("所有 batch 都失败了，无法完成训练。")

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_labels, all_preds, num_classes=2)
    metrics['loss'] = total_loss / num_batches
    return metrics


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(loader):
        try:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = ce_dice_loss(outputs, labels)

            # NaN 检测
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARNING] Val batch {batch_idx}: loss is NaN/Inf, skipping.")
                continue

            total_loss += loss.item()
            num_batches += 1
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"  [ERROR] Val batch {batch_idx}: CUDA OOM. Skipping batch.")
                torch.cuda.empty_cache()
                continue
            raise

    if num_batches == 0:
        raise RuntimeError("所有验证 batch 都失败了，无法完成验证。")

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_labels, all_preds, num_classes=2)
    metrics['loss'] = total_loss / num_batches
    return metrics


# ==================== 主训练流程 ====================

def train(config):
    # 强制 GPU
    assert_gpu()

    # 随机种子
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 加速训练

    device = torch.device('cuda')
    region = config.get('region', 'Shouyang')

    print(f"\n{'='*60}")
    print(f"训练配置")
    print(f"{'='*60}")
    print(f"  地区:       {region}")
    print(f"  模型:       {config['model_name']}")
    print(f"  预训练:     {config.get('pretrained', True)}")
    print(f"  时相数:     {config['num_temporal']}")
    print(f"  波段数:     {config['num_bands']}")
    print(f"  类别数:     2 (耕地/非耕地)")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs:     {config['epochs']}")
    print(f"  LR:          {config['learning_rate']}")
    print(f"  Weight decay: {config.get('weight_decay', 1e-2)}")
    print(f"  Patience:   {config.get('patience', 10)}")
    print(f"  Device:     {device} ({torch.cuda.get_device_name(0)})")
    print(f"{'='*60}\n")

    # 数据集
    data_dir = config['data_dir']
    train_dataset = RemoteSensingDataset(
        data_dir, 'train', config['num_bands'], config['num_temporal'], region)
    val_dataset = RemoteSensingDataset(
        data_dir, 'val', config['num_bands'], config['num_temporal'], region)

    print(f"训练集: {len(train_dataset)} 块")
    print(f"验证集: {len(val_dataset)} 块\n")

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config.get('num_workers', 0), pin_memory=False, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config.get('num_workers', 0), pin_memory=False)

    # 计算类别权重 (耕地占比低时增加权重)
    class_weight = torch.tensor([1.0, 2.0], device=device)

    # 模型（预训练权重）
    model = build_model(
        config['model_name'],
        num_classes=2,
        in_channels=config['num_bands'],
        num_temporal=config['num_temporal'],
        pretrained=config.get('pretrained', True),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params/1e6:.2f}M, 可训练: {trainable_params/1e6:.2f}M\n")

    # 优化器 + 学习率调度
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-2))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'])

    # 训练循环
    best_kappa = -1
    epochs_no_improve = 0
    patience = config.get('patience', 10)
    save_dir = Path(config['save_dir']) / config['experiment_name']
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {'train_loss': [], 'train_kappa': [], 'val_loss': [],
               'val_kappa': [], 'val_oa': [], 'val_miou': [], 'val_f1_crop': [],
               'val_iou_crop': []}

    try:
        for epoch in range(config['epochs']):
            t0 = time.time()
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{config['epochs']}  LR={lr:.6f}")

            try:
                train_metrics = train_one_epoch(model, train_loader, optimizer, device, class_weight)
            except Exception as e:
                print(f"  [ERROR] 训练 epoch {epoch+1} 失败: {e}")
                traceback.print_exc()
                # 保存当前状态以便恢复
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                    'config': config,
                }, save_dir / 'checkpoint_error.pth')
                print(f"  已保存错误状态到 {save_dir / 'checkpoint_error.pth'}")
                raise

            try:
                val_metrics = validate(model, val_loader, device)
            except Exception as e:
                print(f"  [ERROR] 验证 epoch {epoch+1} 失败: {e}")
                traceback.print_exc()
                # 保存当前状态以便恢复
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                    'config': config,
                }, save_dir / 'checkpoint_error.pth')
                print(f"  已保存错误状态到 {save_dir / 'checkpoint_error.pth'}")
                raise

            scheduler.step()

            history['train_loss'].append(train_metrics['loss'])
            history['train_kappa'].append(train_metrics['kappa'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_kappa'].append(val_metrics['kappa'])
            history['val_oa'].append(val_metrics['oa'])
            history['val_miou'].append(val_metrics['miou'])
            history['val_f1_crop'].append(val_metrics['f1_crop'])
            history['val_iou_crop'].append(val_metrics['iou_crop'])

            elapsed = time.time() - t0
            print(f"  Train: loss={train_metrics['loss']:.4f}  kappa={train_metrics['kappa']:.4f}")
            print(f"  Val:   loss={val_metrics['loss']:.4f}  OA={val_metrics['oa']:.4f}  "
                  f"mIoU={val_metrics['miou']:.4f}  kappa={val_metrics['kappa']:.4f}")
            print(f"        IoU_crop={val_metrics['iou_crop']:.4f}  "
                  f"F1_crop={val_metrics['f1_crop']:.4f}  "
                  f"({elapsed:.1f}s)")

            # 保存最佳模型 (按 Kappa )
            if val_metrics['kappa'] > best_kappa:
                best_kappa = val_metrics['kappa']
                epochs_no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_kappa': best_kappa,
                    'val_metrics': val_metrics,
                    'config': config,
                }, save_dir / 'best_model.pth')
                print(f"  -> 保存最佳模型 (Kappa={best_kappa:.4f})")
            else:
                epochs_no_improve += 1
                print(f"  -> 无提升 ({epochs_no_improve}/{patience})")

            # 保存 checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_dir / 'checkpoint_last.pth')

            with open(save_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)

            # Early stopping 检查
            if epochs_no_improve >= patience:
                print(f"\n[Early Stopping] 验证 Kappa 连续 {patience} 轮未提升，训练提前停止。")
                break

    except KeyboardInterrupt:
        print("\n[中断] 用户手动停止训练。")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'config': config,
        }, save_dir / 'checkpoint_interrupted.pth')
        print(f"已保存中断状态到 {save_dir / 'checkpoint_interrupted.pth'}")
        return

    print(f"\n{'='*60}")
    print(f"训练完成! 最佳验证 Kappa: {best_kappa:.4f}")
    print(f"模型保存: {save_dir / 'best_model.pth'}")
    print(f"{'='*60}\n")


# ==================== 入口 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    train(config)


if __name__ == '__main__':
    main()
