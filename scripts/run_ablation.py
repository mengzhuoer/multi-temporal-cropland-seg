"""批量运行 3个模型 x 2个时相 x 2个波段 的12组实验"""
import argparse
import json
from copy import deepcopy
from pathlib import Path


class JSONLLogger:
    """轻量JSONL训练记录替代TensorBoard"""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step, **metrics):
        record = {'step': step, **metrics}
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')


def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(train_fn, config):
    """训练单个配置并返回最佳mIoU"""
    import time
    logger = JSONLLogger(Path(config['log_dir']) / f"{config['experiment_name']}.jsonl")
    set_seed(config['seed'])

    best_iou = 0
    start_time = time.time()

    # 调用实际 train 函数
    model, history = train_fn(config)

    best_iou = max(history['val_iou'])
    result = {
        'experiment': config['experiment_name'],
        'model': config['model_name'],
        'num_temporal': config['num_temporal'],
        'num_bands': config['num_bands'],
        'best_miou': best_iou,
        'training_time_minutes': (time.time() - start_time) / 60,
        'final_train_loss': history['train_loss'][-1],
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config', default='configs/segformer_config.json')
    parser.add_argument('--models', nargs='+', default=['segformer', 'unet', 'deeplabv3plus'])
    parser.add_argument('--output', default='results/ablation_results.json')
    parser.add_argument('--quick_test', action='help',
                        help='临时帮助标志')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖训练轮数（用于快速测试）')
    args = parser.parse_args()

    sys_path = str(Path(__file__).parents[1])
    import sys
    sys.path.insert(0, sys_path)
    from train import load_config, train

    base = load_config(args.base_config)
    if args.epochs is not None:
        base['epochs'] = args.epochs

    model_map = {
        'deeplabv3plus': 'deeplabv3+',
        'deeplabv3': 'deeplabv3+',
    }

    variants = []
    for model in args.models:
        model_name = model_map.get(model.lower(), model.lower())
        for temporal in (6, 1):
            for bands in (6, 3):
                config = deepcopy(base)
                config['model_name'] = model_name
                config['num_temporal'] = temporal
                config['num_bands'] = bands
                config['experiment_name'] = f"{model_name}_t{temporal}_b{bands}"
                variants.append(config)

    print(f"共 {len(variants)} 个实验:")
    for cfg in variants:
        print('  -', cfg['experiment_name'])

    results = []
    for idx, config in enumerate(variants, 1):
        print(f"\n=== 实验 {idx}/{len(variants)}: {config['experiment_name']} ===")
        try:
            result = run_experiment(train, config)
            results.append(result)
        except Exception as e:
            results.append({
                'experiment': config['experiment_name'],
                'error': str(e)
            })
            print(f"训练失败: {e}")

        # 持久化中间结果
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n所有实验完成，结果保存至 {args.output}")


if __name__ == '__main__':
    main()
