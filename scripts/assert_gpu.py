"""强制验证 GPU 可用性, 作为训练前置检查"""
import sys


def assert_gpu(min_cuda_version='12.1'):
    try:
        import torch
    except ImportError:
        raise RuntimeError('未检测到 PyTorch')

    if not torch.cuda.is_available():
        raise RuntimeError('当前环境不可用 CUDA, 量化训练前必须确认驱动和 CUDA 版本')

    version = torch.version.cuda
    print(f'[OK] CUDA {version} / {torch.cuda.get_device_name(0)}')

    # 简单的矩阵乘法验证计算可用性
    x = torch.randn(128, 128, device='cuda')
    y = x @ x
    _ = y.mean().item()
    torch.cuda.synchronize()
    print('[OK] GPU 矩阵计算测试通过')
    return True


if __name__ == '__main__':
    assert_gpu()
