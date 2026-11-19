"""CMPF 主干的单路径渐进跨模态对齐实现。

模型复用基类 ``Model`` 的 Swin、BERT、解码器与损失接口
渐进路径完成多尺度视觉—语言交互。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_model import Model
from model.modules import ASPP, Co_Attention, GatedInjection


# ===========================================
#      单路径渐进跨模态对齐主干
# ===========================================

class CMPFMainBranch(nn.Module):
    """多尺度视觉—语言对齐路径。"""

    def __init__(
        self,
        img_dims=(128, 256, 512, 1024),
        text_dim=768,
        n_heads=8,
        img_sizes=(128, 64, 32, 16),
        dropout=0.1,
        dropout_act=0.0,
    ):
        super().__init__()
        self.drop = nn.Dropout2d(p=dropout)

        self.feature_norms = nn.ModuleList([
            nn.GroupNorm(16, img_dims[0]),
            nn.GroupNorm(32, img_dims[1]),
            nn.GroupNorm(32, img_dims[2]),
            nn.GroupNorm(32, img_dims[3]),
        ])

        # S3 与 S2 各进行两次串行交互。
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

        self.grid_gen_s3 = self._make_grid_gen(img_dims[3], 1, 0, dropout=dropout_act)
        self.gate_s3 = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Sigmoid())
        self.grid_gen_s2 = self._make_grid_gen(img_dims[2], 3, 1, dropout=dropout_act)
        self.gate_s2 = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Sigmoid())
        self.grid_gen_s1 = self._make_grid_gen(img_dims[1], 3, 1, ratio=8, dropout=dropout_act)
        self.gate_s1 = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Sigmoid())

        self.inj_3to2 = GatedInjection(img_dims[3], img_dims[2])
        self.inj_2to1 = GatedInjection(img_dims[2], img_dims[1])
        self.proj_1to0 = nn.Conv2d(img_dims[1], img_dims[0], kernel_size=1)
        self.s0_aspp = ASPP(img_dims[0], img_dims[0], rates=[1, 2, 4])

    @staticmethod
    def _make_grid_gen(in_dim, kernel_size, padding, dropout=0.0, ratio=16):
        return nn.Sequential(
            nn.Conv2d(in_dim, in_dim // ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(in_dim // ratio, 1, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
        )

    @staticmethod
    def _apply_gate(features, grid_generator, gate):
        grid = grid_generator(features)
        modulation = grid * gate(grid)
        return features + features * modulation, modulation

    def forward(self, feats, text_embeds, text_mask=None):
        f0, f1, f2, f3 = feats

        # S3：完整表达式的两次串行交互。
        v3, _ = self.s3_interact_1(f3, text_embeds, text_mask)
        v3, _ = self.s3_interact_2(v3, text_embeds, text_mask)
        v3, g3 = self._apply_gate(v3, self.grid_gen_s3, self.gate_s3)

        # S2：注入高层特征后继续使用同一完整表达式交互。
        f2_injected = self.drop(self.inj_3to2(f2, v3))
        v2, _ = self.s2_interact_1(f2_injected, text_embeds, text_mask)
        v2, _ = self.s2_interact_2(v2, text_embeds, text_mask)
        v2, g2 = self._apply_gate(v2, self.grid_gen_s2, self.gate_s2)

        # S1：高分辨率边界细化。
        f1_injected = self.drop(self.inj_2to1(f1, v2))
        v1, _ = self.s1_interact(f1_injected, text_embeds, text_mask)
        v1, g1 = self._apply_gate(v1, self.grid_gen_s1, self.gate_s1)

        v1_up = F.interpolate(v1, size=f0.shape[2:], mode='bilinear', align_corners=False)
        f0_refined = self.s0_aspp(f0)
        f0_injected = f0_refined + self.proj_1to0(v1_up)

        out_feats = [f0_injected, v1, v2, v3]
        out_feats = [norm(feature) for norm, feature in zip(self.feature_norms, out_feats)]
        return out_feats, [g3, g2, g1]


# ===========================================
#      CMPF 模型
# ===========================================

class CMPFModel(Model):
    """复用基类视觉/文本编码器与解码器，注入 ``CMPFMainBranch``。"""

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
        self.main_branch = CMPFMainBranch(dropout=drop, dropout_act=drop_act)
        self._attention_handles = []
        self._captured_patch_to_token_attention = {}

    def enable_patch_to_token_capture(self):
        """启用图像 patch 查询文本 token 的注意力捕获。

        仅在可视化推理前调用。每个张量的形状为
        ``[batch, image_patches, text_tokens]``，可直接提取 ``car``、``left``
        等 token 对应的 patch 热图。
        """
        self.disable_patch_to_token_capture()
        attention_modules = {
            's3_last': self.main_branch.s3_interact_2.cross_attn_img2txt,
            's2_last': self.main_branch.s2_interact_2.cross_attn_img2txt,
            's1': self.main_branch.s1_interact.cross_attn_img2txt,
        }

        def build_hook(stage_name):
            def hook(_, __, output):
                attention_weights = output[1]
                if attention_weights is not None:
                    self._captured_patch_to_token_attention[stage_name] = attention_weights.detach()
            return hook

        self._captured_patch_to_token_attention.clear()
        self._attention_handles = [
            module.register_forward_hook(build_hook(stage_name))
            for stage_name, module in attention_modules.items()
        ]

    def disable_patch_to_token_capture(self):
        """移除可视化 hook，避免训练阶段保留注意力张量。"""
        for handle in self._attention_handles:
            handle.remove()
        self._attention_handles.clear()
        self._captured_patch_to_token_attention.clear()

    def get_patch_to_token_attention(self):
        """返回最近一次前向传播捕获到的图像 patch→文本 token 权重。"""
        if not self._attention_handles:
            raise RuntimeError('Call enable_patch_to_token_capture() before inference.')
        return dict(self._captured_patch_to_token_attention)

    @staticmethod
    def _flatten_text_inputs(word_id, word_mask):
        if word_id.ndim == 3:
            if word_id.size(1) != 1:
                raise ValueError('CMPFModel expects one full-expression text sequence per sample.')
            word_id = word_id[:, 0]
            word_mask = word_mask[:, 0]
        if word_id.ndim != 2:
            raise ValueError('Text inputs must have shape [batch, tokens] or [batch, 1, tokens].')
        return word_id, word_mask

    def forward(self, image, word_id, word_mask):
        word_id, word_mask = self._flatten_text_inputs(word_id, word_mask)

        visual_features = [
            feature.permute(0, 3, 1, 2).contiguous()
            for feature in self.visual(image)
        ]
        text_embeddings = self.bert(word_id, attention_mask=word_mask).last_hidden_state
        fused_features, gate_maps = self.main_branch(visual_features, text_embeddings)
        output = self.decoder(fused_features)

        if self.training:
            return output
        return output, gate_maps
