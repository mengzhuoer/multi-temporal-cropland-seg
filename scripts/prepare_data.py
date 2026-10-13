"""将 Sentinell-2 多时相影像与30m土地覆盖标签文件格式转换为训练集和验证集。"""
import argparse
import json
from pathlib import Path

import numpy as np


def discover_files(root, extensions=('.tif', '.tiff', '.npy')):
    root = Path(root)
    files = []
    for ext in extensions:
        files.extend(root.rglob(f'*{ext}'))
    return sorted(files)


def load_raster(path):
    suffix = Path(path).suffix.lower()
    if suffix == '.npy':
        return np.load(path)
    try:
        import rasterio
        with rasterio.open(path) as src:
            array = src.read()
            profile = src.profile
        return array, profile
    except ImportError as exc:
        raise ImportError('读取TIF需要 rasterio；请 pip install rasterio') from exc


def validate_stack(stack, label, expected_t=6, expected_bands=6):
    if stack.ndim != 4:
        raise ValueError(f'图像应为(T,C,H,W), 当前维度:{stack.shape}')
    t, c, h, w = stack.shape
    if t != expected_t:
        raise ValueError(f'时相数量错误: 期望{expected_t}, 实际{t}')
    if c not in (3, 6, 10, 13):
        raise ValueError(f'波段数异常: {c}')
    if (c != expected_bands) and expected_bands is not None:
        # 支持从10波段Sentinel-2中筛选6个B2/B3/B4/B8/B11/B12
        pass
    if label.shape != (h, w):
        raise ValueError(f'标签shape {label.shape} 与图像空间shape {(h,w)} 不一致')
    return True


def split_patches(image, label, patch_size=256, stride=256,
                  train_ratio=0.8, seed=42):
    """按空间块划分train/val, 避免同一位置的数据泄漏。"""
    t, c, h, w = image.shape
    positions = []
    for y in range(0, max(1, h - patch_size + 1), stride):
        for x in range(0, max(1, w - patch_size + 1), stride):
            positions.append((y, x))

    rng = np.random.default_rng(seed)
    rng.shuffle(positions)
    cut = int(len(positions) * train_ratio)
    train_pos = positions[:cut]
    val_pos = positions[cut:]
    return train_pos, val_pos


def save_split(image, label, positions, out_dir, prefix):
    out_dir.mkdir(parents=True, exist_ok=True)
    label_dir = out_dir / 'labels'
    label_dir.mkdir(exist_ok=True)
    for idx, (y, x) in enumerate(positions):
        image_path = out_dir / f'{prefix}_{idx:05d}.npy'
        label_path = label_dir / f'{prefix}_{idx:05d}.npy'
        np.save(image_path, image[:, :, y:y+image.shape[-2], x:x+image.shape[-1]])
        np.save(label_path, label[y:y+image.shape[-2], x:x+image.shape[-1]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True, help='文件按地区命名的Sentinel-2影像')
    parser.add_argument('--label_dir', required=True, help='30m土地利用标签')
    parser.add_argument('--output_dir', default='data')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry_run', action='store_true', help='仅检查文件，不写出')
    args = parser.parse_args()

    images = discover_files(args.image_dir)
    labels = discover_files(args.label_dir)
    if len(images) == 0:
        raise RuntimeError(f'未找到影像: {args.image_dir}')
    if len(labels) == 0:
        raise RuntimeError(f'未找到标签: {args.label_dir}')
    if len(images) != len(labels):
        raise RuntimeError(f'图像({len(images)})与标签({len(labels)})数量不匹配')

    metadata = []
    output_root = Path(args.output_dir)
    train_pos, val_pos = None, None
    for i, (image_path, label_path) in enumerate(zip(images), start=1):
        image = load_raster(image_path)
        label = load_raster(label_path)
        validate_stack(image, label)

        if train_pos is None:
            train_pos, val_pos = split_patches(
                image, label, args.patch_size, args.stride, args.train_ratio, args.seed
            )
        if not args.dry_run:
            save_split(image, label, train_pos, output_root / 'train' / 'images', f'region{i:02d}')
            save_split(image, label, val_pos, output_root / 'val' / 'images', f'region{i:02d}')
        metadata.append({
            'region': image_path.stem,
            'image': str(image_path),
            'label': str(label_path),
            'shape': list(image.shape),
            'train_patches': len(train_pos),
            'val_patches': len(val_pos)
        })

    meta_path = output_root / 'split_metadata.json'
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f'完成！{len(meta_path)}个元数据文件，数据写至 {output_root}')


if __name__ == '__main__':
    main()
