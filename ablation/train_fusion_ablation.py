"""编码器与解码器融合策略消融的统一训练入口。"""

import argparse
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation.model.fusion_ablation_model import FUSION_STRATEGIES
from ablation.model.generate_fusion_ablation_model import prepare_fusion_ablation_model
from train import get_args, main


def get_fusion_ablation_args():
    """先解析消融配置，再复用完整表达式训练参数。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ablation-target', choices=('encoder', 'decoder'), required=True)
    parser.add_argument('--fusion-strategy', choices=FUSION_STRATEGIES, required=True)
    ablation_args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_args]
    return get_args(), ablation_args


if __name__ == '__main__':
    args, ablation_args = get_fusion_ablation_args()
    # 八组融合消融仅记录验证指标，不产生最佳或临时权重。
    args.disable_checkpoint = True
    model_factory = partial(
        prepare_fusion_ablation_model,
        ablation_target=ablation_args.ablation_target,
        fusion_strategy=ablation_args.fusion_strategy,
    )
    main(args, model_factory=model_factory)
