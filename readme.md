## CMPF

遥感图像指代分割（Referring Remote Sensing Image Segmentation）方法实现。CMPF 将指代分割
建模为逐级的跨模态过滤：编码端决定哪些文本相关响应被传递到下一尺度，解码端决定各尺度保留
多少空间细节。

## 环境

| 项 | 版本 |
| --- | --- |
| Python | 3.10 |
| torch / torchvision | 2.5.0（CUDA 12.1）/ 0.20.0 |
| transformers / timm | 4.49.0 / 1.0.15 |

```bash
pip install -r requirements.txt
```

- 依赖清单见 `requirements.txt`。
- BERT 权重放在项目内 `./bert-base-uncased/`；Swin-B 由 timm 自动加载。
- 多卡训练使用 `torch.nn.parallel.DistributedDataParallel`，通过 `CUDA_VISIBLE_DEVICES`
  控制可见 GPU。

## 目录结构

```
.
├── train.py                  # 训练入口（CMPF），支持 model_factory 供消融复用
├── test.py                   # 评测入口（CMPF）
├── train.sh / test.sh        # 常用命令
├── model/                    # CMPF 模型实现
│   ├── base_model.py         # 基类 Model、ScaleAdaptiveHead、GatedFusionBlock
│   ├── modules.py            # Co_Attention / ASPP / GroupASPP / GatedInjection / GLU
│   ├── cmpf_model.py         # CMPFMainBranch、CMPFModel
│   ├── cmpf_gated_model.py   # CMPFGatedModel（CMPF 本体）
│   ├── cmpf_decoder.py       # CMPFGatedHead
│   └── build_cmpf_model.py   # 模型构建、权重加载、DDP 与优化器分组
├── ablation/                 # 消融实验入口、模型与批量命令
│   ├── train_fusion_ablation.py            # 编码器/解码器融合策略消融
│   ├── train_coupled_upsample_concat.py    # 上采样拼接消融
│   ├── test_coupled_upsample_concat.py     # 上采样拼接变体评测
│   ├── run_fusion_ablations_rrsisd.sh
│   ├── train_coupled_upsample_concat.sh
│   └── model/
├── dataset/                  # dataloader（ReferDataset / SegmentationDataset）+ transforms.py
├── engine/                   # 训练/验证循环、损失
├── utils/                    # checkpoint、logger、指标、可视化
├── refer/                    # 数据根目录
├── saved_models/             # 权重
├── logs/                     # 训练/评测日志
└── paper_visuals/            # 论文图
```


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
└── RISBench_dataset/
    ├── images/  masks/
    └── output_phrase_{train,val,test}.txt
```

标注文本格式为 `<image_id> <expression>`；图像与掩码统一缩放到 512×512。文本由 BERT
tokenizer 编码为单条完整表达式序列（`max_length=20`）。

## 训练

```bash
# RRSIS-D
python train.py --dataset rrsisd --ngpu 1 --time 17 \
  --savename cmpf_rrsisd --visulize 0 \
  --batch_size 8 --nb_epoch 40 --lr 3e-5

# RISBench
python train.py --dataset risbench --ngpu 1 --time 17 \
  --savename cmpf_risbench --visulize 0 \
  --batch_size 8 --nb_epoch 40 --lr 3e-5
```

| 数据集 | batch | epochs | lr | min_lr |
| --- | --- | --- | --- | --- |
| RRSIS-D | 8 | 40 | 3e-5 | 1e-7 |
| RefSegRS | 4 | 55 | 5e-5 | 2.5e-6 |
| RISBench | 8 | 40 | 3e-5 | 1e-7 |

- DDP 使用当前可见的全部 GPU，`--ngpu` 不生效，用 `CUDA_VISIBLE_DEVICES` 控制。
- 优化器 AdamW（`wd=0.01`，`betas=(0.9,0.98)`），前 3 epoch 线性 warmup 后 cosine 衰减。
- 损失 `0.5 × CE + 0.5 × Dice`，AMP 训练，梯度裁剪 `max_norm=1.0`；BERT 前 8 层冻结。
- 日志写入 `logs/<savename>/log.txt`；权重按数据集命名：
  `saved_models/<dataset>_temp.pth`（含 optimizer）与 `saved_models/<dataset>_best.pth`。
  同一数据集重复训练会互相覆盖，需要保留时请自行备份。

## 评测

```bash
python test.py --eval_splits val test \
  --dataset rrsisd \
  --pretrain saved_models/rrsisd_best.pth \
  --workers 4 --savename eval_rrsisd \
  --bert_tokenizer ./bert-base-uncased/
```

- `test.py` 固定评测 CMPF，没有 `--model_type` 参数。
- `--pretrain` 加载纯 `state_dict`；`--resume` 加载完整 checkpoint 并恢复 optimizer/epoch。
- 评测固定 `batch_size=1`，对每条描述逐条推理。
- 指标：mIoU（逐样本 IoU 均值）、oIoU（全局交并比）、Pr@50/60/70/80/90。
- `--bert_tokenizer` 建议显式传入仓库内 `./bert-base-uncased/`，跨机器时尤其必要。

## 结果

CMPF 评测结果：

| 数据集 | Val mIoU | Test mIoU | Test oIoU | Pr@50 | Pr@70 | Pr@90 |
| --- | --- | --- | --- | --- | --- | --- |
| RRSIS-D | 0.6606 | 0.6477 | 0.7779 | 0.7460 | 0.5780 | 0.2654 |
| RefSegRS | 0.9193 | 0.7901 | 0.8403 | 0.8833 | 0.7672 | 0.4436 |
| RISBench | 0.6955 | 0.7002 | 0.7526 | 0.7718 | 0.6669 | 0.4034 |

## 权重文件
将在接下来会放出