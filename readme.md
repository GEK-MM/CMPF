## CMPF

## 环境

```bash
conda activate pytorch_310
cd /home/ubuntu/glk/code/private-code/any_try/InferSeg/Lab2
```

| 项 | 版本 |
| --- | --- |
| Python | 3.10.15 |
| torch / torchvision | 2.5.0（CUDA 12.1）/ 0.20.0 |
| transformers / timm | 4.49.0 / 1.0.15 |
| GPU | 2 × RTX 3090 24 GB |

- 依赖清单见 `requirements.txt`（部分版本与实际环境不一致，以实际环境为准）。
- BERT 本地权重：项目内 `./bert-base-uncased/`；Swin-B 由 timm 加载。
- 脚本通过 `CUDA_VISIBLE_DEVICES=1` 只使用物理 GPU 1。

## 目录结构

```
Lab2/
├── train.py                  # 训练入口（CMPF），支持 model_factory 供消融复用
├── test.py                   # 评测入口（仅 CMPF）
├── train.sh / test.sh        # 常用命令
├── visualize_rf_progression.py  # RF 逐尺度热力图导出
├── model/                    # CMPF 模型实现（7 个文件）
│   ├── model.py              # 基类 Model、ScaleAdaptiveHead、GatedFusionBlock
│   ├── modules.py            # Co_Attention / ASPP / GroupASPP / GatedInjection / GLU
│   ├── coupled_model.py      # CoupledMainBranch、CoupledModel
│   ├── coupled_gated_decoder_model.py      # CoupledGatedDecoderModel（CMPF）
│   ├── gated_scale_adaptive_decoder.py     # GatedScaleAdaptiveHead
│   └── generate_coupled_*.py # 模型构建、权重加载、DDP 与优化器分组
├── ablation/                 # 消融实验入口、模型与批量命令
│   ├── train_fusion_ablation.py            # 编码器/解码器融合策略消融
│   ├── train_coupled_upsample_concat.py    # 上采样拼接消融
│   ├── test_coupled.py                     # Coupled 变体评测
│   ├── test_coupled_upsample_concat.py     # 上采样拼接变体评测
│   ├── run_fusion_ablations_rrsisd.sh
│   ├── train_coupled_upsample_concat.sh
│   └── model/
├── dataset/                  # dataloader（ReferDataset / SegmentationDataset）
├── engine/                   # 训练/验证循环、损失
├── utils/                    # checkpoint、logger、指标
├── refer/                    # 数据根目录
├── saved_models/             # 权重
├── logs/                     # 训练/评测日志
└── paper_visuals/            # 论文图
```

消融入口会自动把项目根目录加入 `sys.path`，但数据路径相对 cwd，请在项目根目录运行。

## 数据

| 数据集 | 根目录 | 训练 | 验证 | 测试 |
| --- | --- | --- | --- | --- |
| RRSIS-D (`rrsisd`) | `refer/rrsisd-data/` | 12,181 refs | 1,740 refs | 3,481 descs |
| RefSegRS (`refsegrs`) | `refer/RefSegRS/` | 2,172 | 431 | 1,817 |
| RISBench (`risbench`) | `refer/RISBench_dataset/` | 26,299 | 10,012 | 16,158 |

```
refer/
├── refer.py                 # REFER API（RRSIS-D 使用）
├── rrsisd-data/
│   ├── images/
│   └── rrsisd/              # refs(unc).p、instances.json
├── RefSegRS/
│   ├── images/  masks/      # .tif
│   └── output_phrase_{train,val,test}.txt
├── RISBench_dataset/
│   ├── images/  masks/
│   └── output_phrase_{train,val,test}.txt
├── RefSe.zip
└── RISBench_dataset.zip
```

标注文本格式为 `<image_id> <expression>`；图像与掩码统一缩放到 512×512。文本由 BERT tokenizer
编码为单条完整表达式序列（`max_length=20`）。

## 训练

```bash
# CMPF，RISBench（train.sh 当前内容）
export CUDA_VISIBLE_DEVICES=1
python train.py --dataset risbench --ngpu 1 --time 17 \
  --savename coupled_gated_decoder_risbench --visulize 0 \
  --batch_size 8 --nb_epoch 40 --lr 3e-5

# RRSIS-D
python train.py --dataset rrsisd --ngpu 1 --time 17 \
  --savename coupled_gated_decoder_rrsisd --visulize 0 \
  --batch_size 8 --nb_epoch 40 --lr 3e-5
```

| 数据集 | batch | epochs | lr | min_lr |
| --- | --- | --- | --- | --- |
| RRSIS-D | 8 | 40 | 3e-5 | 1e-7 |
| RefSegRS | 4 | 55 | 5e-5 | 2.5e-6 |
| RISBench | 8 | 40 | 3e-5 | 1e-7 |

- DDP 使用当前可见的全部 GPU，`--ngpu` 实际不生效，用 `CUDA_VISIBLE_DEVICES` 控制。
- 优化器 AdamW（`wd=0.01`，`betas=(0.9,0.98)`），前 3 epoch 线性 warmup 后 cosine 衰减。
- 损失 `0.5 × CE + 0.5 × Dice`，AMP 训练，梯度裁剪 `max_norm=1.0`；BERT 前 8 层冻结。
- 日志写入 `logs/<savename>/log.txt`；权重按数据集命名：
  `saved_models/<dataset>_temp.pth`（含 optimizer）与 `saved_models/<dataset>_best.pth`。
  **同一数据集重复训练会互相覆盖**，需要保留时手动移动。

## 评测

```bash
python test.py --eval_splits val test \
  --dataset rrsisd \
  --pretrain saved_models/couple_decoder/co_de_rrsisd_best.pth \
  --workers 4 --savename final_eval_rrsisd \
  --bert_tokenizer /home/ubuntu/glk/code/bert-base-uncased/
```

- `test.py` 固定评测 CMPF，没有 `--model_type` 参数。
- `--pretrain` 加载纯 `state_dict`；`--resume` 加载完整 checkpoint 并恢复 optimizer/epoch。
- 评测固定 `batch_size=1`，对每条描述逐条推理。
- 指标：mIoU（逐样本 IoU 均值）、oIoU（全局交并比）、Pr@50/60/70/80/90。
- `test.py` 的 `--bert_tokenizer` 默认指向 `/home/ubuntu/glk/code/bert-base-uncased/`，与
  `train.py` 的 `./bert-base-uncased/` 不同，跨机器时显式传入。

## 消融实验

```bash
# 8 组融合策略消融（编码器/解码器 × sum/mul/gated/concat_conv），RRSIS-D，25 epochs
bash ablation/run_fusion_ablations_rrsisd.sh

# 上采样拼接消融（RRSIS-D，40 epochs；RefSegRS / RISBench 命令在脚本内注释）
bash ablation/train_coupled_upsample_concat.sh

# 单独运行某组融合消融
python ablation/train_fusion_ablation.py --ablation-target encoder --fusion-strategy sum ...
```

| 入口 | 模型 |
| --- | --- |
| `ablation/train_fusion_ablation.py` | `CoupledEncoderFusionAblationModel` / `CoupledDecoderFusionAblationModel` |
| `ablation/train_coupled_upsample_concat.py` | `CoupledUpsampleConcatModel`（RF+SFI → 上采样拼接） |
| `ablation/test_coupled.py` | `CoupledModel`（CMPF 去掉解码器门控） |
| `ablation/test_coupled_upsample_concat.py` | `CoupledUpsampleConcatModel` |

`train_fusion_ablation.py` 不保存权重（`disable_checkpoint=True`），只记录指标。

## 可视化

```bash
python visualize_rf_progression.py \
  --weights saved_models/couple_decoder/co_de_rrsisd_best.pth \
  --output-dir paper_visuals/rf_progression \
  --dataset rrsisd --data-root ./refer \
  --bert-tokenizer /home/ubuntu/glk/code/bert-base-uncased/ \
  --min-iou 0.85 --max-outputs 12
```

- 导出 5 列图：原图 + RF4 16×16 + RF3 32×32 + RF2 64×64 + 预测/GT。
- `--response-mode relative` 每个尺度独立归一化；`--hide-text` 去掉文字与标题。
- 训练期对比图（`--visulize 1`）输出到 `test_sample/epoch_<n>/`。

## 结果

CMPF 最终评测（`logs/final_evals/*.log`）：

| 数据集 | Val mIoU | Test mIoU | Test oIoU | Pr@50 | Pr@70 | Pr@90 | 权重 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RRSIS-D | 0.6606 | 0.6477 | 0.7779 | 0.7460 | 0.5780 | 0.2654 | `saved_models/couple_decoder/co_de_rrsisd_best.pth` |
| RefSegRS | 0.9193 | 0.7901 | 0.8403 | 0.8833 | 0.7672 | 0.4436 | `saved_models/couple_decoder/coupe_de_refsegrs_best.pth` |
| RISBench | 0.6955 | 0.7002 | 0.7526 | 0.7718 | 0.6669 | 0.4034 | `saved_models/couple_decoder/coupe_de_risbench_best.pth` |

DDPA 基线（历史指标，代码已删除）：Test mIoU RRSIS-D 0.6468、RefSegRS 0.7866、RISBench 0.6956。
消融结果见 `.Agent/MEMORY/04-experiments.md`。

## 备注

- `logs/` 只保留 CMPF、Coupled、消融与最终评测日志；DDPA 及更早期（CLIP / refcoco 时代）的
  日志已于 2026-11-04 删除。
- 根目录 `args.py` 为早期 RMSIN/LAVT 代码遗留，未被任何模块引用。
- `engine/length_eval.py` 提供按描述长度分桶评测，目前只在 `test.py` 中 import，未接入主流程。
