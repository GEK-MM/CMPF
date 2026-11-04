"""普通上采样拼接卷积消融模型的构建与优化器配置。"""

import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ablation.model.coupled_upsample_concat_model import CoupledUpsampleConcatModel
from utils.checkpoint import load_pretrain


def prepare_coupled_upsample_concat_model(args, rank, logger):
    """构建独立消融模型，优化器参数分组与耦合基线保持一致。"""
    model = CoupledUpsampleConcatModel(
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
