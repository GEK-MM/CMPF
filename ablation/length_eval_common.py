"""按指代表达式长度分桶评测的公共实现（RefSegRS / RISBench 共用）。

长度取原始文本的单词数，不受 BERT 20 个 token 截断影响；超过 20 词的样本
单独归入 [21,+]。
"""

import json
import os
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = '/home/ubuntu/glk/code/private-code/any_try/InferSeg/Lab2'
sys.path.insert(0, PROJECT_ROOT)

from model.coupled_gated_decoder_model import CoupledGatedDecoderModel
from train import get_dataset, get_transform

BINS = [(1, 2), (3, 5), (6, 9), (10, 20), (21, 10 ** 9)]
LABELS = ['[1,2]', '[3,5]', '[6,9]', '[10,20]', '[21,+]']
PREC_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def build_args(dataset, data_root, bert_tokenizer, img_size, num_query):
    return SimpleNamespace(
        data_root=data_root,
        dataset=dataset,
        refer_data_root=os.path.join(data_root, 'rrsisd-data'),
        bert_tokenizer=bert_tokenizer,
        img_size=img_size,
        num_query=num_query,
        splitBy='unc',
        lr=3e-5,
        drop_act=0.12,
    )


def load_model(weights, bert_tokenizer, img_size, num_query, device):
    model = CoupledGatedDecoderModel(
        tunelang=True,
        num_query=num_query,
        img_size=img_size,
        text_model_path=bert_tokenizer,
        drop=0.1,
        drop_act=0.12,
    ).to(device)
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {key.removeprefix('module.'): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def bucket_index(length):
    for index, (low, high) in enumerate(BINS):
        if low <= length <= high:
            return index
    return len(BINS) - 1


def _empty_stats():
    return {
        'count': 0,
        'inter': 0.0,
        'union': 0.0,
        'iou_sum': 0.0,
        'prec': {t: 0 for t in PREC_THRESHOLDS},
    }


def _report(stats):
    if stats['count'] == 0:
        return None
    return {
        'count': stats['count'],
        'mIoU': stats['iou_sum'] / stats['count'],
        'oIoU': stats['inter'] / (stats['union'] + 1e-6),
        **{f'Pr@{int(t * 100)}': stats['prec'][t] / stats['count'] for t in PREC_THRESHOLDS},
    }


def evaluate_split(split, dataset_args, model, device, workers=4):
    dataset, _ = get_dataset(split, get_transform(dataset_args), dataset_args)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=workers, pin_memory=True,
    )
    buckets = [_empty_stats() for _ in LABELS]

    with torch.no_grad():
        for index, (images, masks, word_ids, word_masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)
            word_ids = word_ids.to(device)
            word_masks = word_masks.to(device)
            if word_ids.ndim == 4:
                word_ids = word_ids[:, :, :, 0]
                word_masks = word_masks[:, :, :, 0]

            logits, _ = model(images, word_ids, word_masks)
            prediction = logits.argmax(dim=1)
            intersection = torch.logical_and(prediction == 1, masks == 1).sum().item()
            union = torch.logical_or(prediction == 1, masks == 1).sum().item()
            iou = intersection / (union + 1e-6)

            length = len(dataset.data[index]['text'].split())
            stats = buckets[bucket_index(length)]
            stats['count'] += 1
            stats['inter'] += intersection
            stats['union'] += union
            stats['iou_sum'] += iou
            for t in PREC_THRESHOLDS:
                stats['prec'][t] += 1 if iou > t else 0

            if index % 1000 == 0:
                print(f'[{split} {index}/{len(dataset)}] iou={iou:.4f}', flush=True)

    report = {label: _report(stats) for label, stats in zip(LABELS, buckets)}
    count = sum(stats['count'] for stats in buckets)
    report['ALL'] = {
        'count': count,
        'mIoU': sum(stats['iou_sum'] for stats in buckets) / max(count, 1),
        'oIoU': sum(stats['inter'] for stats in buckets)
        / (sum(stats['union'] for stats in buckets) + 1e-6),
    }
    return report


def run(dataset, weights, splits, data_root, bert_tokenizer, img_size, num_query, output):
    device = torch.device('cuda:0')
    dataset_args = build_args(dataset, data_root, bert_tokenizer, img_size, num_query)
    model = load_model(weights, bert_tokenizer, img_size, num_query, device)

    results = {}
    for split in splits:
        results[split] = evaluate_split(split, dataset_args, model, device)
        print(f'=== {dataset} {split} ===', flush=True)
        for label in LABELS + ['ALL']:
            row = results[split][label]
            if row is None:
                print(f'{label:>8} | null', flush=True)
            else:
                print(f'{label:>8} | n={row["count"]:5d} | mIoU={row["mIoU"]:.4f} '
                      f'| oIoU={row["oIoU"]:.4f}', flush=True)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(f'saved {output}', flush=True)
