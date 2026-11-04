import os
import sys
import argparse
import random
import datetime
import matplotlib as mpl

from train import get_dataset, get_transform

mpl.use('Agg')
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data.distributed
from torch.cuda.amp import autocast as autocast
from engine.engine import *
from dataset.data_loader import *
from utils.losses import *
from utils.parsing_metrics import *
from utils.utils import *
from utils.checkpoint import load_resume
from utils.logger import setup_logger
from model.generate_coupled_gated_decoder_model import prepare_coupled_gated_decoder_model
from engine.length_eval import validate_epoch_by_length_bins

def get_args():
    parser = argparse.ArgumentParser(description='CMPF evaluation')
    parser.add_argument('--gpu', default='2', help='gpu id')
    parser.add_argument('--ngpu', default=2, type=int, help='gpu num')
    parser.add_argument('--workers', default=4, type=int, help='num workers for data loading')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    parser.add_argument('--nb_epoch', default=32, type=int, help='training epoch')
    parser.add_argument('--lr', default=0.000025, type=float, help='batch size 16 learning rate')
    parser.add_argument('--power', default=0.1, type=float, help='lr poly power')
    parser.add_argument('--steps', default=[15, 28], type=list, help='in which step lr decay by power')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--dataset', default='rrsisd', type=str, )
    parser.add_argument('--eval_splits', nargs='+', default=['val', 'test'],
                        choices=['val', 'test'],
                        help='dataset splits to evaluate in order')
    parser.add_argument('--img_size', default=512, type=int, help='image size')
    parser.add_argument('--drop_fusion', default=0.1, help='dropout for fusion')
    parser.add_argument('--drop_act', default=0, help='dropout for activate')

    parser.add_argument('--num_query', default=20, type=int, help='the number of query')
    parser.add_argument('--w_seg', default=0.1, type=float, help='weight of the seg loss')
    parser.add_argument('--w_coord', default=5, type=float, help='weight of the reg loss')
    parser.add_argument('--tunelang', dest='tunelang', default=True, action='store_true',
                        help='if finetune language model')

    parser.add_argument('--time', default=15, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='path to ReferIt splits data folder')

    parser.add_argument('--fusion_dim', default=768, type=int,
                        help='fusion module embedding dimensions')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                        help='pretrain support load state_dict that are not identical, while have no loss saved as resume')
    parser.add_argument('--print_freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 1e3)')
    parser.add_argument('--savename', default='default', type=str, help='Name head for saved model')

    parser.add_argument('--seg_thresh', default=0.35, type=float, help='seg score above this value means foreground')
    parser.add_argument('--seg_out_stride', default=2, type=int, help='the seg out stride')
    parser.add_argument('--best_iou', default=-float('Inf'), type=int, help='the best accu')
    parser.add_argument('--visulize', default=0, type=int, help='visulize of picture')

    parser.add_argument('--data_root', default='./refer', help='Root directory for all datasets')
    parser.add_argument('--refer_data_root', default='./refer/rrsisd-data/', help='REFER dataset root directory')
    parser.add_argument('--split', default='test', help='only used when testing')
    parser.add_argument('--splitBy', default='unc',
                        help='change to umd or google when the datasset is G-Ref (RefCOCOg)')
    parser.add_argument('--bert_tokenizer', default='/home/ubuntu/glk/code/bert-base-uncased/', help='BERT tokenizer')

    global args, anchors_full, writer, logger
    args = parser.parse_args()
    args.gsize = 32
    args.date = datetime.datetime.now().strftime('%Y%m%d')
    if args.savename == 'default':
        args.savename = 'model_v1_%s_batch%d_%s' % (args.dataset, args.batch_size, args.date)
    os.makedirs(args.log_dir, exist_ok=True)
    args.lr = args.lr * (args.batch_size * args.ngpu // 16)

    print('----------------------------------------------------------------------')
    print(sys.argv[0])
    print(args)
    print('----------------------------------------------------------------------')

    return args


def main(args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12367'

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print("Running DDP with {} GPUs".format(n_gpus))
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, args,))
    else:
        print("Please use GPU for evaluation")


def run(rank, n_gpus, args):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.cuda.set_device(rank)

    ## fix seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 2)
    torch.cuda.manual_seed_all(args.seed + 3)

    ## save logs
    logger = setup_logger(output=os.path.join(args.log_dir, args.savename), distributed_rank=rank, color=False,
                          name="cmpf")
    logger.info(str(sys.argv))
    logger.info(str(args))

    ## Model
    model, optimizer = prepare_coupled_gated_decoder_model(args, rank, logger)

    args.start_epoch = 0
    if args.resume:
        model = load_resume(model, optimizer, args, logger, rank)
        model.to(rank)
        best_miou_seg = args.best_iou
        print(best_miou_seg)

    for split in args.eval_splits:
        dataset, _ = get_dataset(split, get_transform(args=args), args=args)
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=args.workers,
        )
        if rank == 0:
            validate_epoch(args, data_loader, model, logger, split.title(), 0)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    args = get_args()
    main(args)
