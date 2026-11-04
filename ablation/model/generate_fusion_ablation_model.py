"""融合策略消融模型的构建与优化器配置。"""

import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ablation.model.fusion_ablation_model import (
    CoupledDecoderFusionAblationModel,
    CoupledEncoderFusionAblationModel,
    FUSION_STRATEGIES,
)
from utils.checkpoint import load_pretrain


def prepare_fusion_ablation_model(args, rank, logger, ablation_target, fusion_strategy):
    """构建只替换指定融合模块的独立消融模型。"""
    if fusion_strategy not in FUSION_STRATEGIES:
        raise ValueError(f'Unsupported fusion strategy: {fusion_strategy}.')

    model_classes = {
        'encoder': CoupledEncoderFusionAblationModel,
        'decoder': CoupledDecoderFusionAblationModel,
    }
    if ablation_target not in model_classes:
        raise ValueError(f'Unsupported ablation target: {ablation_target}.')

    model = model_classes[ablation_target](
        fusion_strategy=fusion_strategy,
        tunelang=args.tunelang,
        num_query=args.num_query,
        img_size=args.img_size,
        text_model_path=args.bert_tokenizer,
        drop=args.drop_fusion,
        drop_act=args.drop_act,
    ).cuda(rank)

    if args.pretrain and os.path.isfile(args.pretrain):
        model = load_pretrain(model, args, logger, rank)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model_without_ddp = model.module
    visual_parameters = [
        parameter for name, parameter in model_without_ddp.named_parameters()
        if name.startswith('visual') and parameter.requires_grad
    ]
    text_parameters = [
        parameter for name, parameter in model_without_ddp.named_parameters()
        if name.startswith('bert') and parameter.requires_grad
    ]
    other_parameters = [
        parameter for name, parameter in model_without_ddp.named_parameters()
        if parameter.requires_grad
        and not name.startswith('visual')
        and not name.startswith('bert')
    ]
    optimizer = torch.optim.AdamW(
        [
            {'params': other_parameters, 'lr': args.lr},
            {'params': visual_parameters, 'lr': args.lr},
            {'params': text_parameters, 'lr': args.lr},
        ],
        weight_decay=0.01,
        betas=(0.9, 0.98),
    )
    return model, optimizer
