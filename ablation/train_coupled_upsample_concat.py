"""普通上采样拼接卷积消融实验的独立训练入口。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation.model.generate_coupled_upsample_concat_model import prepare_coupled_upsample_concat_model
from train import get_args, main


if __name__ == '__main__':
    args = get_args()
    if args.savename.startswith('coupled_'):
        args.savename = args.savename.replace('coupled_', 'coupled_upsample_concat_', 1)
    main(args, model_factory=prepare_coupled_upsample_concat_model)
