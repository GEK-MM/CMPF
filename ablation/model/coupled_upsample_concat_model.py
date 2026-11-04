"""使用普通上采样拼接卷积的完整表达式耦合消融模型。

该模型保留 CoupledModel 的编码器、五次跨模态交互和门控尺度自适应解码器，
仅将主分支中的 RF 调制与 SFI 替换为普通的逐级上采样、特征拼接和卷积融合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.coupled_model import CoupledModel
from model.gated_scale_adaptive_decoder import GatedScaleAdaptiveHead
from model.modules import Co_Attention


# ===========================================
#      普通跨尺度上采样拼接融合
# ===========================================

class UpsampleConcatConv(nn.Module):
    """将深层特征上采样后与当前尺度特征拼接，再用卷积融合。"""

    def __init__(self, low_channels, high_channels, out_channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(low_channels + high_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, low_features, high_features):
        high_features = F.interpolate(
            high_features,
            size=low_features.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        return self.fusion(torch.cat([low_features, high_features], dim=1))


# ===========================================
#      无 RF 与 SFI 的主分支
# ===========================================

class UpsampleConcatMainBranch(nn.Module):
    """以普通逐级拼接卷积替换 RF+SFI 的完整表达式主分支。"""

    def __init__(
        self,
        img_dims=(128, 256, 512, 1024),
        text_dim=768,
        n_heads=8,
        img_sizes=(128, 64, 32, 16),
        dropout=0.1,
    ):
        super().__init__()
        self.drop = nn.Dropout2d(p=dropout)
        self.feature_norms = nn.ModuleList([
            nn.GroupNorm(16, img_dims[0]),
            nn.GroupNorm(32, img_dims[1]),
            nn.GroupNorm(32, img_dims[2]),
            nn.GroupNorm(32, img_dims[3]),
        ])

        # 与耦合基线保持相同的五次跨模态交互，确保仅消融 RF+SFI。
        self.s3_interact_1 = Co_Attention(
            img_dims[3], text_dim, n_heads, img_sizes[3], img_sizes[3],
            use_self_attn=True, use_aspp=True, use_group_aspp=True, groups=4,
            rate=[1, 3, 5], expansion_factor=1.5, use_pos=True,
        )
        self.s3_interact_2 = Co_Attention(
            img_dims[3], text_dim, n_heads, img_sizes[3], img_sizes[3],
            use_self_attn=True, use_aspp=True, use_group_aspp=True, groups=4,
            rate=[1, 2, 3], expansion_factor=1.5, use_pos=True,
        )
        self.s2_interact_1 = Co_Attention(
            img_dims[2], text_dim, n_heads, img_sizes[2], img_sizes[2],
            use_self_attn=False, use_aspp=True, use_group_aspp=True, groups=4,
            rate=[1, 3, 5], expansion_factor=1.5, use_pos=True,
        )
        self.s2_interact_2 = Co_Attention(
            img_dims[2], text_dim, n_heads, img_sizes[2], img_sizes[2],
            use_self_attn=False, use_aspp=True, use_group_aspp=True, groups=4,
            rate=[1, 2, 3], expansion_factor=1.5, use_pos=True,
        )
        self.s1_interact = Co_Attention(
            img_dims[1], text_dim, n_heads, img_sizes[1], img_sizes[1],
            use_self_attn=False, use_aspp=True, use_group_aspp=True, groups=4,
            rate=[1, 2, 3, 5], expansion_factor=2.0, use_pos=True,
        )

        self.fuse_3to2 = UpsampleConcatConv(img_dims[2], img_dims[3], img_dims[2])
        self.fuse_2to1 = UpsampleConcatConv(img_dims[1], img_dims[2], img_dims[1])
        self.fuse_1to0 = UpsampleConcatConv(img_dims[0], img_dims[1], img_dims[0])

    def forward(self, feats, text_embeds, text_mask=None):
        f0, f1, f2, f3 = feats

        v3, _ = self.s3_interact_1(f3, text_embeds, text_mask)
        v3, _ = self.s3_interact_2(v3, text_embeds, text_mask)

        f2_fused = self.drop(self.fuse_3to2(f2, v3))
        v2, _ = self.s2_interact_1(f2_fused, text_embeds, text_mask)
        v2, _ = self.s2_interact_2(v2, text_embeds, text_mask)

        f1_fused = self.drop(self.fuse_2to1(f1, v2))
        v1, _ = self.s1_interact(f1_fused, text_embeds, text_mask)

        f0_fused = self.drop(self.fuse_1to0(f0, v1))
        out_feats = [f0_fused, v1, v2, v3]
        out_feats = [norm(feature) for norm, feature in zip(self.feature_norms, out_feats)]
        return out_feats, []


# ===========================================
#      独立消融模型
# ===========================================

class CoupledUpsampleConcatModel(CoupledModel):
    """保留门控尺度自适应解码器，仅以普通拼接卷积替换主分支 RF+SFI。"""

    def __init__(self, tunelang=False, num_query=20, length=17,
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
        self.main_branch = UpsampleConcatMainBranch(dropout=drop)
        self.decoder = GatedScaleAdaptiveHead(
            target_size=(img_size, img_size),
            dropout=drop,
            gate_dropout=drop_act,
        )
