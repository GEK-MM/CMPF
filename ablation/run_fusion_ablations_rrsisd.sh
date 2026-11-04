#!/usr/bin/env bash
# RRSIS-D 融合策略消融：GPU 1 上顺序运行 8 组实验。

set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1

run_case() {
  local target="$1"
  local strategy="$2"
  local run_name="fusion_${target}_${strategy}_rrsisd"

  echo "===== Starting ${run_name} ====="
  python ablation/train_fusion_ablation.py \
    --ablation-target "$target" \
    --fusion-strategy "$strategy" \
    --dataset rrsisd \
    --ngpu 1 \
    --time 17 \
    --savename "$run_name" \
    --visulize 0 \
    --batch_size 8 \
    --nb_epoch 25 \
    --lr 3e-5

  echo "===== Finished ${run_name} ====="
}

# 编码器融合：仅替换 SFI，固定 CMA 和 RSF。
for strategy in sum mul gated concat_conv; do
  run_case encoder "$strategy"
done

# 解码器融合：仅替换 RSF，固定 CMA、SFI 和解码器空间门控。
for strategy in sum mul gated concat_conv; do
  run_case decoder "$strategy"
done
