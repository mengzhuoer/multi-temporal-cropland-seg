"""使用训练好的30m模型推理10m分辨率图像

用法:
    python scripts/predict_10m.py \
        --weights results/checkpoints/best_model.pth \
        --images data/10m_region1/ \
        --output predictions/
        --model_name segformer \
        --num_bands 6 --num_temporal 6
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model_name', choices=['unet', 'segformer', 'deeplabv3+'], default='segformer')
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_bands', type=int, default=6)
    parser.add_argument('--num_temporal', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


if __name__ == '__main__':
    args = load_config()
    print(f"使用权重: {args.weights}")
    print(f"模型: {args.model_name}, 波段: {args.num_bands}, 时相: {args.num_temporal}")

    # 这里需要根据实际数据格式补充推理逻辑
    print("请先将10m影像制备为(6, 6, H, W) .npy文件后运行此脚本")
