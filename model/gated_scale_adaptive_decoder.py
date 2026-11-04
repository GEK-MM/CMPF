"""带渐进空间门控的多尺度解码器。

保留原 ``ScaleAdaptiveHead`` 的特征投影、逐尺度融合与分类头，只在由深到浅
完成每一级融合后施加一次空间门控。该文件独立存在，不改变原解码器。
"""

import torch.nn as nn
import torch.nn.functional as F

from model.model import GatedFusionBlock


# ===========================================
#      解码阶段的空间门控
# ===========================================

class DecoderSpatialGate(nn.Module):
    """复用主干 ``_apply_gate`` 的调制形式。"""

    def __init__(self, channels, dropout=0.0, ratio=16):
        super().__init__()
        self.grid_generator = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(channels // ratio, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        grid = self.grid_generator(features)
        modulation = grid * self.gate(grid)
        return features + features * modulation


# ===========================================
#      自底向上的门控多尺度解码器
# ===========================================

class GatedScaleAdaptiveHead(nn.Module):
    """在每次尺度融合后使用空间门控的解码器。

    Args:
        in_channels: 四个输入尺度的通道数，顺序由高分辨率到低分辨率。
        embedding_dim: 统一的解码通道数。
        target_size: 最终掩码尺寸。
        dropout: 与原解码器保持一致的投影和分类 Dropout。
        gate_dropout: 空间门控内部的 Dropout 概率。
    """

    def __init__(
        self,
        in_channels=(128, 256, 512, 1024),
        embedding_dim=512,
        target_size=(512, 512),
        dropout=0.1,
        gate_dropout=0.0,
    ):
        super().__init__()
        self.target_size = target_size
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, embedding_dim, kernel_size=1),
                nn.GroupNorm(32, embedding_dim),
                nn.Dropout2d(p=dropout),
            )
            for channels in in_channels
        ])
        self.semantic_anchor = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, embedding_dim),
            nn.GELU(),
            nn.Dropout2d(p=dropout),
        )
        self.fusion_blocks = nn.ModuleList([
            GatedFusionBlock(embedding_dim) for _ in range(3)
        ])
        # 每次由低分辨率语义向高分辨率特征融合后，都重新估计空间门控。
        self.decoder_gates = nn.ModuleList([
            DecoderSpatialGate(embedding_dim, dropout=gate_dropout)
            for _ in range(3)
        ])
        self.classifier = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(16, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout2d(p=0.15),
            nn.Conv2d(embedding_dim // 2, 2, kernel_size=1),
        )

    def forward(self, inputs):
        f0, f1, f2, f3 = [
            projection(feature) for projection, feature in zip(self.projs, inputs)
        ]

        # 从最深特征开始，逐级向高分辨率融合并门控。
        current = self.semantic_anchor(f3)
        for high_resolution, fusion_block, spatial_gate in zip(
            (f2, f1, f0), self.fusion_blocks, self.decoder_gates
        ):
            current = fusion_block(high_resolution, current)
            current = spatial_gate(current)

        logits = self.classifier(current)
        return F.interpolate(
            logits,
            size=self.target_size,
            mode='bilinear',
            align_corners=False,
        )
