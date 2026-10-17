"""Trainable reconstruction of the released rec.onnx computation graph.

Depthwise Conv+BN pairs were fused by the original exporter. We retain those
fused convolutions as trainable parameters; the unpublished original training
architecture, optimizer and batch statistics cannot be recovered exactly.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

FORMAT = "indra-rec-reconstructed-v1"


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, hidden, 1)
        self.fc2 = nn.Conv1d(hidden, channels, 1)

    def forward(self, x, mask):
        pooled = (x * mask).sum(2, keepdim=True) / mask.sum(2, keepdim=True).clamp_min(1)
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(pooled))))
        return x * gate


class Block(nn.Module):
    def __init__(self, channels, expansion, se_hidden, kernel):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels)
        self.expand = nn.Conv1d(channels, expansion, 1)
        self.se = SqueezeExcitation(expansion, se_hidden)
        self.project = nn.Conv1d(expansion, channels, 1)

    def forward(self, x, mask):
        y = F.hardswish(self.expand(self.dw(x)))
        return (x + self.project(self.se(y, mask))) * mask


class Recognizer(nn.Module):
    def __init__(self, classes: int, height: int = 32, channels: int = 128,
                 expansion: int = 256, se_hidden: int = 64, blocks: int = 10,
                 kernel: int = 9, stride: int = 4, bn_eps: float = 1e-5):
        super().__init__()
        self.config = dict(classes=classes, height=height, channels=channels,
                           expansion=expansion, se_hidden=se_hidden, blocks=blocks,
                           kernel=kernel, stride=stride, bn_eps=bn_eps)
        self.stem = nn.Conv2d(1, channels, (height, stride), stride=(height, stride))
        self.stem_bn = nn.BatchNorm1d(channels, eps=bn_eps)
        self.blocks = nn.ModuleList(
            [Block(channels, expansion, se_hidden, kernel) for _ in range(blocks)]
        )
        self.head = nn.Conv1d(channels, classes, 1)

    def forward(self, x, widths):
        x = self.stem(x).squeeze(2)
        stride = self.config["stride"]
        lengths = torch.div(widths + stride - 1, stride, rounding_mode="floor").clamp(max=x.shape[2])
        mask = (torch.arange(x.shape[2], device=x.device)[None, :] < lengths[:, None])[:, None]
        mask = mask.to(x.dtype)
        x = F.hardswish(self.stem_bn(x)) * mask
        for block in self.blocks:
            x = block(x, mask)
        return self.head(x).transpose(1, 2), lengths

    def resize_classes(self, classes: int):
        if classes < self.head.out_channels:
            raise ValueError("vocabulary may only be extended")
        previous = self.head
        head = nn.Conv1d(previous.in_channels, classes, 1).to(previous.weight)
        with torch.no_grad():
            head.weight[:previous.out_channels].copy_(previous.weight)
            head.bias[:previous.out_channels].copy_(previous.bias)
            # Avoid new tokens overwhelming the pretrained classifier at startup.
            head.bias[previous.out_channels:].fill_(-5.0)
        self.head = head
        self.config["classes"] = classes


class Inference(nn.Module):
    """The three-output contract expected by indra_ocr.pipeline.OrtOCR."""
    def __init__(self, model):
        super().__init__()
        self.rec = model

    def forward(self, x, widths):
        logits, lengths = self.rec(x, widths)
        ids = logits.argmax(-1)
        confidence = logits.softmax(-1).amax(-1)
        return ids, confidence, lengths


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_checkpoint(path, model, meta, source, **extra):
    checkpoint = dict(format=FORMAT, config=dict(model.config), state_dict=model.state_dict(),
                      meta=meta, source=source, **extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(cpu_tree(checkpoint), temporary)
    temporary.replace(path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("not a reconstructed Indra recognizer checkpoint")
    model = Recognizer(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, checkpoint
