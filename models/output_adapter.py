"""统一不同模型的输出格式为 Tensor (B, C, H, W)。"""
import torch
import torch.nn as nn


class OutputAdapter(nn.Module):
    """将不同模型输出封装为统一 Tensor。

    - DeepLabV3+: OrderedDict -> 取 'out'
    - SegFormer: ModelOutput -> 取 .logits
    - UNet: Tensor -> 直接返回
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            # torchvision segmentation models
            if "out" in out:
                return out["out"]
            raise RuntimeError(f"未知dict输出格式: {out.keys()}")
        elif hasattr(out, "logits"):
            # HuggingFace Segformer / 其他 transformers 模型
            return out.logits
        elif isinstance(out, torch.Tensor):
            return out
        else:
            raise RuntimeError(f"不支持的模型输出类型: {type(out)}")
