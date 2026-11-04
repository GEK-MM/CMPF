"""普通上采样拼接卷积消融实验的独立评估入口。"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from ablation.model.generate_coupled_upsample_concat_model import prepare_coupled_upsample_concat_model
from engine.engine import validate_epoch
from test import get_args as get_test_args
from train import get_dataset, get_transform
from utils.checkpoint import load_resume
from utils.logger import setup_logger


def main(args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'
    if not torch.cuda.is_available():
        raise RuntimeError('Upsample-concat ablation evaluation requires CUDA.')

    n_gpus = torch.cuda.device_count()
    print(f'Running upsample-concat ablation evaluation with {n_gpus} GPU(s)')
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, args))


def run(rank, n_gpus, args):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.cuda.set_device(rank)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logger = setup_logger(
        output=os.path.join(args.log_dir, f'coupled_upsample_concat_{args.savename}'),
        distributed_rank=rank,
        color=False,
        name='coupled-upsample-concat',
    )
    val_dataset, _ = get_dataset('val', get_transform(args), args)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=args.workers,
    )

    model, optimizer = prepare_coupled_upsample_concat_model(args, rank, logger)
    if args.resume:
        model = load_resume(model, optimizer, args, logger, rank)

    if rank == 0:
        validate_epoch(args, val_loader, model.module, logger, 'Val', 0)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main(get_test_args())
