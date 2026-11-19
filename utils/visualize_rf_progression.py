"""导出 Coupled-Gated 模型的逐尺度 RF 定位热力图。

仅用于论文可视化：不改模型、不写入检查点，也不参与训练。输出顺序固定为
RF4 (16×16) → RF3 (32×32) → RF2 (64×64)，用于观察指代响应由粗到细的演化。
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.cmpf_gated_model import CMPFGatedModel
from train import get_dataset, get_transform


# ===========================================
#      参数与模型加载
# ===========================================

def parse_args():
    parser = argparse.ArgumentParser(description='Export progressive RF heatmaps.')
    parser.add_argument('--weights', required=True, help='Coupled-Gated 权重路径。')
    parser.add_argument('--output-dir', default='paper_visuals/rf_progression')
    parser.add_argument('--dataset', default='rrsisd', choices=['rrsisd'])
    parser.add_argument('--data-root', default='./refer')
    parser.add_argument('--bert-tokenizer', default='/home/ubuntu/glk/code/bert-base-uncased/')
    parser.add_argument('--img-size', default=512, type=int)
    parser.add_argument('--num-query', default=20, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--min-iou', default=0.85, type=float)
    parser.add_argument('--max-outputs', default=12, type=int)
    parser.add_argument(
        '--hide-text', action='store_true',
        help='不绘制面板标题、表达式或 IoU 文本，仅保留一行五列图像。',
    )
    parser.add_argument(
        '--response-mode', choices=['positive', 'relative'], default='positive',
        help='positive 使用统一正增强色阶；relative 对每一尺度的调制响应独立归一化。',
    )
    return parser.parse_args()


def build_dataset_args(args):
    """构造 ReferDataset 所需的最小参数集合。"""
    return SimpleNamespace(
        data_root=args.data_root,
        dataset=args.dataset,
        refer_data_root=os.path.join(args.data_root, 'rrsisd-data'),
        bert_tokenizer=args.bert_tokenizer,
        img_size=args.img_size,
        num_query=args.num_query,
        splitBy='unc',
        lr=3e-5,
        drop_act=0.12,
    )


def load_model(args, device):
    """严格加载当前 CMA–RF–SFI + RSF 架构的训练权重。"""
    model = CMPFGatedModel(
        tunelang=True,
        num_query=args.num_query,
        img_size=args.img_size,
        text_model_path=args.bert_tokenizer,
        drop=0.1,
        drop_act=0.12,
    ).to(device)
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {
        key.removeprefix('module.'): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# ===========================================
#      热力图与结果绘制
# ===========================================

def compute_iou(logits, target):
    prediction = logits.argmax(dim=1)
    target = target.long()
    intersection = torch.logical_and(prediction == 1, target == 1).sum().float()
    union = torch.logical_or(prediction == 1, target == 1).sum().float()
    return float(intersection / (union + 1e-6))


def denormalize_image(image):
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
    return torch.clamp(image * std + mean, 0, 1).permute(1, 2, 0).cpu().numpy()


def safe_name(text):
    compact = re.sub(r'[^a-zA-Z0-9]+', '_', text.lower()).strip('_')
    return compact[:72] or 'expression'


def token_ids_to_text(word_ids, tokenizer):
    ids = word_ids.detach().cpu().tolist()
    if isinstance(ids[0], list):
        ids = ids[0]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def render_progression(
    image, target, logits, gate_maps, expression, iou, output_path,
    response_mode, hide_text=False,
):
    """将三个 RF 响应叠加到同一原图上。"""
    image_np = denormalize_image(image)
    height, width = image_np.shape[:2]

    raw_responses = [gate[0, 0].cpu().numpy() for gate in gate_maps]
    if response_mode == 'positive':
        # 正响应代表增强；负响应代表抑制。
        responses = [np.maximum(response, 0.0) for response in raw_responses]
        color_max = max(max(float(response.max()) for response in responses), 1e-6)
        stage_labels = ['RF4 · 16×16', 'RF3 · 32×32', 'RF2 · 64×64']
    else:
        # 每层独立归一化，只展示该层内部的空间响应分布。
        responses = []
        for response in raw_responses:
            low, high = float(response.min()), float(response.max())
            responses.append((response - low) / max(high - low, 1e-6))
        color_max = 1.0
        stage_labels = ['Relative RF4 · 16×16', 'Relative RF3 · 32×32', 'Relative RF2 · 64×64']
    layout = {'wspace': 0.008} if hide_text else None
    fig, axes = plt.subplots(
        1, 5, figsize=(16, 3.4), constrained_layout=not hide_text,
        gridspec_kw=layout,
    )
    axes[0].imshow(image_np)
    if not hide_text:
        axes[0].set_title('Image', fontsize=12)

    for axis, response, label in zip(axes[1:4], responses, stage_labels):
        resized = torch.nn.functional.interpolate(
            torch.from_numpy(response)[None, None],
            size=(height, width),
            mode='bilinear',
            align_corners=False,
        )[0, 0].numpy()
        axis.imshow(image_np)
        axis.imshow(resized, cmap='jet', vmin=0.0, vmax=color_max,
                    alpha=0.52, interpolation='bilinear')
        if not hide_text:
            axis.set_title(label, fontsize=12)

    prediction = logits[0].argmax(dim=0).detach().cpu().numpy() == 1
    target_np = target.detach().cpu().numpy() == 1
    axes[4].imshow(image_np)
    overlay = np.zeros_like(image_np)
    overlay[prediction] = [0.0, 1.0, 0.0]
    axes[4].imshow(overlay, alpha=0.32)
    if target_np.any():
        axes[4].contour(target_np, levels=[0.5], colors='red', linewidths=1.8)
    if not hide_text:
        axes[4].set_title(f'Prediction / GT\nIoU {iou:.3f}', fontsize=12)

    for axis in axes:
        axis.axis('off')
    if not hide_text:
        fig.suptitle(expression, fontsize=12, y=1.02)
    fig.savefig(
        output_path,
        dpi=240,
        bbox_inches='tight',
        pad_inches=0 if hide_text else 0.03,
    )
    plt.close(fig)


# ===========================================
#      验证集筛选与导出
# ===========================================

def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this visualization script.')

    device = torch.device('cuda:0')
    dataset_args = build_dataset_args(args)
    dataset, _ = get_dataset('val', get_transform(dataset_args), dataset_args)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=args.workers,
    )
    model = load_model(args, device)
    tokenizer = dataset.tokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []

    with torch.no_grad():
        for index, (images, masks, word_ids, word_masks) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            word_ids = word_ids.to(device, non_blocking=True)
            word_masks = word_masks.to(device, non_blocking=True)
            # ReferDataset 在验证集保留描述索引 [B, 1, tokens, descriptions]。
            # Coupled 模型每次只处理一条完整表达式；RRSIS-D 验证集此处 descriptions=1。
            if word_ids.ndim == 4:
                word_ids = word_ids[..., 0]
                word_masks = word_masks[..., 0]
            logits, gate_maps = model(images, word_ids, word_masks)
            iou = compute_iou(logits, masks)
            if iou < args.min_iou:
                continue

            expression = token_ids_to_text(word_ids[0], tokenizer)
            filename = f'{len(metadata) + 1:02d}_{safe_name(expression)}_iou_{iou:.3f}.png'
            render_progression(
                images[0], masks[0], logits, gate_maps, expression, iou,
                output_dir / filename, args.response_mode, args.hide_text,
            )
            metadata.append({
                'dataset_index': index,
                'expression': expression,
                'iou': round(iou, 6),
                'file': filename,
                'gate_sizes': [list(gate.shape[-2:]) for gate in gate_maps],
                'response_mode': args.response_mode,
                'hide_text': args.hide_text,
            })
            print(f'[{len(metadata)}/{args.max_outputs}] {filename}')
            if len(metadata) >= args.max_outputs:
                break

    with (output_dir / 'metadata.json').open('w', encoding='utf-8') as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f'Exported {len(metadata)} figures to {output_dir}')


if __name__ == '__main__':
    main()
