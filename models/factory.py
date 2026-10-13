"""模型构建工厂：支持预训练权重、6/3波段自适应。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .output_adapter import OutputAdapter


class TemporalBandReshape(nn.Module):
    """将 (B, T, C, H, W) 重塑为 (B, T*C, H, W)。"""
    def forward(self, x):
        B, T, C, H, W = x.shape
        return x.reshape(B, T * C, H, W)


def adapt_first_stage_conv(conv_layer: nn.Conv2d, total_input_channels: int) -> nn.Conv2d:
    """适配第一个stage的patch embedding卷积层。"""
    if total_input_channels == 3:
        return conv_layer
    num_copies = total_input_channels // 3
    old_w = conv_layer.weight.data
    new_w = old_w.repeat(1, num_copies, 1, 1) / num_copies
    new_conv = nn.Conv2d(
        total_input_channels, conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size, stride=conv_layer.stride,
        padding=conv_layer.padding, bias=conv_layer.bias is not None,
    )
    new_conv.weight.data = new_w
    if conv_layer.bias is not None:
        new_conv.bias.data = conv_layer.bias.data
    return new_conv


class SegFormerMultiTemporal(nn.Module):
    """5D 输入 -> 4D 重塑 -> SegFormer -> logits。"""
    def __init__(self, segformer_model, num_temporal=1, num_bands=6):
        super().__init__()
        self.reshape = TemporalBandReshape()
        self.segformer = segformer_model
        self.num_temporal = num_temporal
        self.num_bands = num_bands

    def forward(self, x):
        B, T, C, H, W = x.shape
        # 单时相时只取前3个波段，兼容预训练权重
        if T == 1 and C > 3:
            x = x[:, :, :3, :, :]
        x = x.reshape(B, T * min(C, 3) if T == 1 else T * C, H, W)
        out = self.segformer(x)
        logits = out.logits if hasattr(out, "logits") else out
        if logits.shape[-2] != H or logits.shape[-1] != W:
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits


class DeepLabV3PlusMultiTemporal(nn.Module):
    """5D 输入 -> 4D 重塑 -> DeepLabV3+ -> logits tensor。

    aux_loss=True 时模型返回 dict{"out", "aux"}，此处只取 "out"。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        out = self.model(x)
        if isinstance(out, dict):
            out = out["out"]
        if out.shape[-2] != H or out.shape[-1] != W:
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out


def build_segformer(num_classes: int, in_channels: int, num_temporal: int,
                    pretrained: bool = True) -> nn.Module:
    try:
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
    except ImportError:
        raise RuntimeError("请安装transformers: pip install transformers")

    # 单时相时只取前3个波段，使 total_channels=3 兼容预训练权重
    if num_temporal == 1:
        total_channels = 3
    else:
        total_channels = in_channels * num_temporal

    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=num_classes, ignore_mismatched_sizes=True,
        )
        if total_channels != 3:
            stages = list(model.segformer.stages)
            first_stage = stages[0]
            old_proj = first_stage.patch_embeddings.proj
            first_stage.patch_embeddings.proj = adapt_first_stage_conv(old_proj, total_channels)
        return SegFormerMultiTemporal(model, num_temporal=num_temporal, num_bands=in_channels)
    else:
        config = SegformerConfig(
            num_channels=total_channels, num_labels=num_classes,
            num_encoder_blocks=4, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            hidden_sizes=[32, 64, 160, 256],
            num_attention_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
        )
        base_model = SegformerForSemanticSegmentation(config)
        return SegFormerMultiTemporal(base_model, num_temporal=num_temporal, num_bands=in_channels)


def build_deeplabv3plus(num_classes: int, in_channels: int, num_temporal: int,
                        pretrained: bool = True) -> nn.Module:
    """构建DeepLabV3+：先以21类加载预训练(aux_loss=True)，再替换分类头和首层卷积。

    torchvision要求预训练时aux_loss=True(预训练权重含辅助分类头)，
    wrapper中只取dict的"out"字段用于训练。
    """
    try:
        from torchvision.models.segmentation import (
            deeplabv3_resnet101, DeepLabV3_ResNet101_Weights,
        )
    except ImportError:
        raise RuntimeError("请安装torchvision")

    total_channels = in_channels * num_temporal

    if pretrained:
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        model = deeplabv3_resnet101(weights=weights, num_classes=21, aux_loss=True)
    else:
        model = deeplabv3_resnet101(weights=None, num_classes=num_classes, aux_loss=False)

    # 替换主分类头
    old_classifier = model.classifier[-1]
    model.classifier[-1] = nn.Conv2d(
        old_classifier.in_channels, num_classes, kernel_size=1
    )

    # 替换辅助分类头（保持结构一致）
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        old_aux = model.aux_classifier[-1]
        model.aux_classifier[-1] = nn.Conv2d(
            old_aux.in_channels, num_classes, kernel_size=1
        )

    # 替换第一层卷积适配多时相多波段
    old_conv = model.backbone.conv1
    old_weights = old_conv.weight.data
    new_conv = nn.Conv2d(
        total_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size, stride=old_conv.stride,
        padding=old_conv.padding, bias=old_conv.bias is not None,
    )
    if total_channels == 3:
        new_conv.weight.data = old_weights
    else:
        num_copies = total_channels // 3
        new_conv.weight.data = old_weights.repeat(1, num_copies, 1, 1) / num_copies
    if old_conv.bias is not None:
        new_conv.bias.data = old_conv.bias.data
    model.backbone.conv1 = new_conv

    return DeepLabV3PlusMultiTemporal(model)


class UNetWithEncoder(nn.Module):
    """U-Net：预训练ResNet50编码器 + 跳跃连接解码器。"""
    def __init__(self, encoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.enc_channels = [64, 256, 512, 1024, 2048]
        self.dec_channels = [256, 128, 64, 32]

        self.up5 = self._make_up(self.enc_channels[4], self.dec_channels[0], self.enc_channels[3])
        self.up4 = self._make_up(self.dec_channels[0], self.dec_channels[1], self.enc_channels[2])
        self.up3 = self._make_up(self.dec_channels[1], self.dec_channels[2], self.enc_channels[1])
        self.up2 = self._make_up(self.dec_channels[2], self.dec_channels[3], self.enc_channels[0])
        self.up1 = self._make_up(self.dec_channels[3], 64, self.enc_channels[0])

        self.classifier = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    @staticmethod
    def _make_up(in_ch: int, out_ch: int, skip_ch: int):
        conv_t = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        bn_t = nn.BatchNorm2d(out_ch)
        relu_t = nn.ReLU(inplace=True)
        conv1 = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)
        bn1 = nn.BatchNorm2d(out_ch)
        relu1 = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        bn2 = nn.BatchNorm2d(out_ch)
        relu2 = nn.ReLU(inplace=True)
        return nn.ModuleDict({
            'conv_t': conv_t, 'bn_t': bn_t, 'relu_t': relu_t,
            'conv1': conv1, 'bn1': bn1, 'relu1': relu1,
            'conv2': conv2, 'bn2': bn2, 'relu2': relu2,
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)

        e1 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        e2 = self.encoder.layer1(self.encoder.maxpool(e1))
        e3 = self.encoder.layer2(e2)
        e4 = self.encoder.layer3(e3)
        e5 = self.encoder.layer4(e4)

        d = self.up5['conv_t'](e5)
        d = self.up5['bn_t'](d)
        d = self.up5['relu_t'](d)
        d = F.interpolate(d, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d = torch.cat([d, e4], dim=1)
        d = self.up5['conv1'](d)
        d = self.up5['bn1'](d)
        d = self.up5['relu1'](d)
        d = self.up5['conv2'](d)
        d = self.up5['bn2'](d)
        d = self.up5['relu2'](d)

        d = self.up4['conv_t'](d)
        d = self.up4['bn_t'](d)
        d = self.up4['relu_t'](d)
        d = F.interpolate(d, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d = torch.cat([d, e3], dim=1)
        d = self.up4['conv1'](d)
        d = self.up4['bn1'](d)
        d = self.up4['relu1'](d)
        d = self.up4['conv2'](d)
        d = self.up4['bn2'](d)
        d = self.up4['relu2'](d)

        d = self.up3['conv_t'](d)
        d = self.up3['bn_t'](d)
        d = self.up3['relu_t'](d)
        d = F.interpolate(d, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d = torch.cat([d, e2], dim=1)
        d = self.up3['conv1'](d)
        d = self.up3['bn1'](d)
        d = self.up3['relu1'](d)
        d = self.up3['conv2'](d)
        d = self.up3['bn2'](d)
        d = self.up3['relu2'](d)

        d = self.up2['conv_t'](d)
        d = self.up2['bn_t'](d)
        d = self.up2['relu_t'](d)
        d = F.interpolate(d, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d = torch.cat([d, e1], dim=1)
        d = self.up2['conv1'](d)
        d = self.up2['bn1'](d)
        d = self.up2['relu1'](d)
        d = self.up2['conv2'](d)
        d = self.up2['bn2'](d)
        d = self.up2['relu2'](d)

        d = self.up1['conv_t'](d)
        d = self.up1['bn_t'](d)
        d = self.up1['relu_t'](d)
        if e1.shape[-2] != d.shape[-2] or e1.shape[-1] != d.shape[-1]:
            e1_aligned = F.interpolate(e1, size=d.shape[-2:], mode="bilinear", align_corners=False)
        else:
            e1_aligned = e1
        d = torch.cat([d, e1_aligned], dim=1)
        d = self.up1['conv1'](d)
        d = self.up1['bn1'](d)
        d = self.up1['relu1'](d)
        d = self.up1['conv2'](d)
        d = self.up1['bn2'](d)
        d = self.up1['relu2'](d)

        return self.classifier(d)


def build_unet_encoder(num_classes: int, in_channels: int, num_temporal: int,
                       pretrained: bool = True) -> nn.Module:
    try:
        from torchvision.models import resnet50, ResNet50_Weights
    except ImportError:
        raise RuntimeError("请安装torchvision")

    total_channels = in_channels * num_temporal
    if pretrained:
        encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        encoder = resnet50(weights=None)

    old_conv = encoder.conv1
    new_conv = nn.Conv2d(
        total_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size, stride=old_conv.stride,
        padding=old_conv.padding, bias=old_conv.bias is not None,
    )
    if total_channels == 3:
        new_conv.weight.data = old_conv.weight.data
    else:
        num_copies = total_channels // 3
        new_conv.weight.data = old_conv.weight.data.repeat(1, num_copies, 1, 1) / num_copies
    if old_conv.bias is not None:
        new_conv.bias.data = old_conv.bias.data
    encoder.conv1 = new_conv
    return UNetWithEncoder(encoder, num_classes)


def build_model(model_name: str, num_classes: int, in_channels: int,
                num_temporal: int, pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "segformer":
        model = build_segformer(num_classes, in_channels, num_temporal, pretrained)
    elif model_name in ("deeplabv3+", "deeplabv3plus"):
        model = build_deeplabv3plus(num_classes, in_channels, num_temporal, pretrained)
    elif model_name == "unet":
        model = build_unet_encoder(num_classes, in_channels, num_temporal, pretrained)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    return OutputAdapter(model)
