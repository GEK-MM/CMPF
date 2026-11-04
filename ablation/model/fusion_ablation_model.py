"""编码器与解码器融合策略的独立消融模型。

该文件只封装融合策略替换，不修改既有 CMA、SFI 或 RSF 实现。编码器消融仅替换
CoupledMainBranch 中的两处 SFI，解码器消融仅替换 GatedScaleAdaptiveHead 中的
三处 RSF；未被消融的模块保持原实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.coupled_gated_decoder_model import CoupledGatedDecoderModel
from model.coupled_model import CoupledMainBranch
from model.gated_scale_adaptive_decoder import GatedScaleAdaptiveHead


# ===========================================
#      跨尺度融合策略
# ===========================================

FUSION_STRATEGIES = ('sum', 'mul', 'gated', 'concat_conv')


class CrossScaleFusion(nn.Module):
    """跨尺度融合基类，负责深层特征对齐与投影。"""

    def __init__(self, low_channels, high_channels):
        super().__init__()
        self.high_projection = nn.Conv2d(high_channels, low_channels, kernel_size=1)

    def _align_features(self, low_features, high_features):
        high_features = F.interpolate(
            high_features,
            size=low_features.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        return low_features, self.high_projection(high_features)


class ElementwiseSumFusion(CrossScaleFusion):
    """逐元素相加融合。"""

    def forward(self, low_features, high_features):
        low_features, high_features = self._align_features(low_features, high_features)
        return low_features + high_features


class ElementwiseMulFusion(CrossScaleFusion):
    """逐元素相乘融合。"""

    def forward(self, low_features, high_features):
        low_features, high_features = self._align_features(low_features, high_features)
        return low_features * high_features


class SimpleGatedFusion(CrossScaleFusion):
    """以 GAP 和二路 Softmax 权重进行的无细化门控融合。

    该策略按 ``alpha * low + beta * high`` 输出，不包含 RSF 特有的
    3×3 卷积、批归一化和 GELU 细化块。
    """

    def __init__(self, low_channels, high_channels):
        super().__init__(low_channels, high_channels)
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(low_channels * 2, 2, kernel_size=1),
            nn.Softmax(dim=1),
        )

    def forward(self, low_features, high_features):
        low_features, high_features = self._align_features(low_features, high_features)
        weights = self.weight_generator(torch.cat([low_features, high_features], dim=1))
        return low_features * weights[:, 0:1] + high_features * weights[:, 1:2]


class ConcatConvFusion(CrossScaleFusion):
    """通道拼接后以 3×3 卷积细化的融合策略。"""

    def __init__(self, low_channels, high_channels):
        super().__init__(low_channels, high_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(low_channels * 2, low_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(low_channels),
            nn.GELU(),
        )

    def forward(self, low_features, high_features):
        low_features, high_features = self._align_features(low_features, high_features)
        return self.fusion(torch.cat([low_features, high_features], dim=1))


def build_cross_scale_fusion(strategy, low_channels, high_channels):
    """按策略名称创建跨尺度融合模块。"""
    fusion_classes = {
        'sum': ElementwiseSumFusion,
        'mul': ElementwiseMulFusion,
        'gated': SimpleGatedFusion,
        'concat_conv': ConcatConvFusion,
    }
    if strategy not in fusion_classes:
        raise ValueError(f'Unsupported fusion strategy: {strategy}.')
    return fusion_classes[strategy](low_channels, high_channels)


# ===========================================
#      编码器 SFI 消融
# ===========================================

class EncoderFusionAblationBranch(CoupledMainBranch):
    """保留 CMA 与 S0 路径，仅替换两处 SFI。"""

    def __init__(self, fusion_strategy, img_dims=(128, 256, 512, 1024), **kwargs):
        super().__init__(img_dims=img_dims, **kwargs)
        self.inj_3to2 = build_cross_scale_fusion(
            fusion_strategy,
            low_channels=img_dims[2],
            high_channels=img_dims[3],
        )
        self.inj_2to1 = build_cross_scale_fusion(
            fusion_strategy,
            low_channels=img_dims[1],
            high_channels=img_dims[2],
        )


class CoupledEncoderFusionAblationModel(CoupledGatedDecoderModel):
    """仅消融编码器 SFI，保留 CMA 和完整门控解码器。"""

    def __init__(self, fusion_strategy, tunelang=False, num_query=20, length=17,
                 img_size=406, text_model_path='', drop=0.1, drop_act=0.0):
        super().__init__(
            tunelang=tunelang,
            num_query=num_query,
            length=length,
            img_size=img_size,
            text_model_path=text_model_path,
            drop=drop,
            drop_act=drop_act,
        )
        self.main_branch = EncoderFusionAblationBranch(
            fusion_strategy=fusion_strategy,
            dropout=drop,
            dropout_act=drop_act,
        )


# ===========================================
#      解码器 RSF 消融
# ===========================================

class DecoderFusionAblationHead(GatedScaleAdaptiveHead):
    """保留解码器空间门控，仅替换三处 RSF 融合单元。"""

    def __init__(self, fusion_strategy, in_channels=(128, 256, 512, 1024),
                 embedding_dim=512, target_size=(512, 512), dropout=0.1,
                 gate_dropout=0.0):
        super().__init__(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            target_size=target_size,
            dropout=dropout,
            gate_dropout=gate_dropout,
        )
        self.fusion_blocks = nn.ModuleList([
            build_cross_scale_fusion(
                fusion_strategy,
                low_channels=embedding_dim,
                high_channels=embedding_dim,
            )
            for _ in range(3)
        ])


class CoupledDecoderFusionAblationModel(CoupledGatedDecoderModel):
    """仅消融解码器 RSF，保留 CMA、SFI 和解码器空间门控。"""

    def __init__(self, fusion_strategy, tunelang=False, num_query=20, length=17,
                 img_size=406, text_model_path='', drop=0.1, drop_act=0.0):
        super().__init__(
            tunelang=tunelang,
            num_query=num_query,
            length=length,
            img_size=img_size,
            text_model_path=text_model_path,
            drop=drop,
            drop_act=drop_act,
        )
        self.decoder = DecoderFusionAblationHead(
            fusion_strategy=fusion_strategy,
            target_size=(img_size, img_size),
            dropout=drop,
            gate_dropout=drop_act,
        )
