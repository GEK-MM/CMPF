#!/usr/bin/env bash
# 普通上采样拼接卷积消融：固定使用物理 GPU 1。

set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1

# 当前实验：RRSIS-D。
python ablation/train_coupled_upsample_concat.py \
  --dataset rrsisd \
  --ngpu 1 \
  --time 17 \
  --savename coupled_upsample_concat_rrsisd \
  --visulize 0 \
  --batch_size 8 \
  --nb_epoch 40 \
  --lr 3e-5

# RefSegRS：batch_size 4，训练 55 轮，学习率 5e-5。
# python ablation/train_coupled_upsample_concat.py --dataset refsegrs --ngpu 1 --time 17 --savename coupled_upsample_concat_refsegrs --visulize 0 --batch_size 4 --nb_epoch 55 --lr 5e-5

# RISBench：batch_size 8，训练 40 轮，学习率 3e-5。
# python ablation/train_coupled_upsample_concat.py --dataset risbench --ngpu 1 --time 17 --savename coupled_upsample_concat_risbench --visulize 0 --batch_size 8 --nb_epoch 40 --lr 3e-5
