"""完整表达式耦合对照模型的构建与优化器配置。"""

import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from model.coupled_model import CoupledModel
from utils.checkpoint import load_pretrain


def prepare_coupled_model(args, rank, logger):
    model = CoupledModel(
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

    visual_params = [
        parameter for name, parameter in model_without_ddp.named_parameters()
        if name.startswith('visual') and parameter.requires_grad
    ]
    text_params = [
        parameter for name, parameter in model_without_ddp.named_parameters()
        if name.startswith('bert') and parameter.requires_grad
    ]
    remaining_params = [
        parameter for name, parameter in model_without_ddp.named_parameters()
        if parameter.requires_grad and not name.startswith('visual') and not name.startswith('bert')
    ]
    optimizer = torch.optim.AdamW(
        [
            {'params': remaining_params, 'lr': args.lr},
            {'params': visual_params, 'lr': args.lr},
            {'params': text_params, 'lr': args.lr},
        ],
        weight_decay=0.01,
        betas=(0.9, 0.98),
    )
    return model, optimizer
