"""在验证集上评估训练好的模型（完整指标）。

用法:
    python evaluate.py \
        --config configs/exp_shouyang_segformer_6t6b.json \
        --weights outputs/checkpoints/Shouyang_SegFormer_6T6B/best_model.pth
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from models.factory import build_model
from utils.metrics import compute_metrics


class RemoteSensingDataset(Dataset):
    """遥感图像分割数据集（支持按空间划分 train/val）。"""

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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = self.labels_dir / img_path.name.replace('_img', '_lbl')
        if not lbl_path.exists():
            lbl_path = self.labels_dir / img_path.name

        image = np.load(img_path)  # (T, C, H, W)
        label = np.load(lbl_path)  # (H, W)

        if self.num_bands == 3:
            image = image[:, :3, :, :]
        if self.num_temporal == 1:
            image = image[0:1, :, :, :]

        image = torch.from_numpy(image.copy()).float()
        label = torch.from_numpy(label.copy()).long()
        return image, label


def evaluate(model, loader, device):
    """在验证集上评估，逐 patch 计算指标。"""
    model.eval()
    all_preds, all_labels_arr = [], []
    total_loss = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels_arr.append(labels.cpu().numpy())

    elapsed = time.time() - t0

    all_preds = np.concatenate(all_preds)
    all_labels_arr = np.concatenate(all_labels_arr)
    metrics = compute_metrics(all_labels_arr, all_preds, num_classes=2)
    metrics['loss'] = total_loss / max(len(loader), 1)
    metrics['elapsed'] = elapsed
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--weights', required=True, help='模型权重路径')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = json.load(f)

    device = torch.device('cuda')
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})\n")

    model = build_model(
        config['model_name'], num_classes=2,
        in_channels=config['num_bands'],
        num_temporal=config['num_temporal'],
        pretrained=False,
    ).to(device)

    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    if state_dict and next(iter(state_dict.keys())).startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print(f"权重加载成功: {args.weights}\n")

    val_dataset = RemoteSensingDataset(
        config['data_dir'], 'val', config['num_bands'], config['num_temporal'], config['region'])
    val_loader = DataLoader(
        val_dataset, batch_size=config.get('batch_size', 2), shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    print(f"验证集: {len(val_dataset)} 块")
    print(f"images_dir: {val_dataset.images_dir}")
    print(f" Num images: {len(val_dataset.images)}\n")

    metrics = evaluate(model, val_loader, device)

    print("=" * 60)
    print(f"{'Experiment':<30} {config.get('experiment_name', '')}")
    print(f"{'Region':<30} {config['region']}")
    print(f"{'Model':<30} {config['model_name']}")
    print(f"{'Bands x Temporal':<30} {config['num_bands']} x {config['num_temporal']}")
    print("=" * 60)
    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    print(f"{'Loss':<20} {metrics['loss']:>10.4f}")
    print(f"{'OA':<20} {metrics['oa']:>10.4f}")
    print(f"{'mIoU':<20} {metrics['miou']:>10.4f}")
    print(f"{'Kappa':<20} {metrics['kappa']:>10.4f}")
    print(f"{'F1 (crop)':<20} {metrics['f1_crop']:>10.4f}")
    print(f"{'IoU (crop)':<20} {metrics['iou_crop']:>10.4f}")
    print(f"\n耗时: {metrics['elapsed']:.1f}s")


if __name__ == '__main__':
    main()
