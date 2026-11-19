"""CMPF（门控解码器）的模型与优化器构建。"""

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from model.cmpf_gated_model import CMPFGatedModel
from utils.checkpoint import load_pretrain


def prepare_cmpf_model(args, rank, logger):
    """构建独立模型，参数分组与 CMPF 保持一致。"""
    model = CMPFGatedModel(
        tunelang=args.tunelang,
        num_query=args.num_query,
        img_size=args.img_size,
        text_model_path=args.bert_tokenizer,
        drop=args.drop_fusion,
        drop_act=args.drop_act,
    ).cuda(rank)

    if args.pretrain:
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
