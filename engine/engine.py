import os
import time
import warnings

import cv2
import matplotlib as mpl
import torch.nn.parallel
import torch.optim
from torch.cuda.amp import autocast as autocast
from transformers import BertTokenizer

from dataset.data_loader import *
from engine.loss import Loss
from utils.losses import *
from utils.parsing_metrics import *
from utils.utils import *
from utils.utils import dice_loss, sigmoid_focal_loss
import random

mpl.use('Agg')
warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
from matplotlib import cm, pyplot as plt
import numpy as np
import torch.nn.functional as F


def word_ids_to_string(word_ids_list, tokenizer):
    """
    将批量样本的token IDs转换回字符串
    word_ids_list: 2D list of token IDs, e.g., [[101, 2054, 102, 0], [101, 1996, 102, 0]]
    """
    texts = []

    for word_ids in word_ids_list:
        # 移除padding和特殊token
        valid_tokens = [id for id in word_ids if id not in [0, 101, 102]]

        # 转换回tokens
        tokens = tokenizer.convert_ids_to_tokens(valid_tokens)

        # 转换回字符串
        text = tokenizer.convert_tokens_to_string(tokens)
        texts.append(text)
    return " ".join(texts)


class AverageMeter(object):
    """用于存储和计算平均值及当前值的类"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def criterion(input, target, weight=0.1, epoch=0):
    return Loss(weight=weight)(input, target, epoch)


def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)
    return iou, intersection, union


import os
import time
import warnings

import cv2
import matplotlib as mpl
import torch.nn.parallel
import torch.optim
from torch.cuda.amp import autocast as autocast
from transformers import BertTokenizer

from dataset.data_loader import *
from engine.loss import Loss
from utils.losses import *
from utils.parsing_metrics import *
from utils.utils import *
from utils.utils import dice_loss, sigmoid_focal_loss
import random

mpl.use('Agg')
warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
from matplotlib import cm, pyplot as plt
import numpy as np
import torch.nn.functional as F


def word_ids_to_string(word_ids_list, tokenizer):
    """
    将批量样本的token IDs转换回字符串
    word_ids_list: 2D list of token IDs, e.g., [[101, 2054, 102, 0], [101, 1996, 102, 0]]
    """
    texts = []

    for word_ids in word_ids_list:
        # 移除padding和特殊token
        valid_tokens = [id for id in word_ids if id not in [0, 101, 102]]

        # 转换回tokens
        tokens = tokenizer.convert_ids_to_tokens(valid_tokens)

        # 转换回字符串
        text = tokenizer.convert_tokens_to_string(tokens)
        texts.append(text)
    return " ".join(texts)


def criterion(input, target, weight=0.1, epoch=0):
    return Loss(weight=weight)(input, target, epoch)


def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)
    return iou, intersection, union


def compute_batch_iou(pred, gt):
    """
    Args:
        pred (Tensor): [B, 2, H, W] - 模型的原始输出
        gt (Tensor): [B, H, W] - 标签
    Returns:
        total_inter (float): 当前 batch 所有的交集像素数
        total_union (float): 当前 batch 所有的并集像素数
        ious (Tensor): 每个样本的 IoU 数组，用于计算 mIoU 和 Precision@X
    """
    # 转换预测结果 [B, H, W]
    pred_class = pred.argmax(dim=1)

    # 将 pred 和 gt 展平为 [B, N] 方便计算
    pred_flat = pred_class.view(pred_class.size(0), -1)
    gt_flat = gt.view(gt.size(0), -1)

    # 计算交集和并集 (针对类别 1)
    intersection = torch.logical_and(pred_flat == 1, gt_flat == 1).sum(dim=1).float()
    union = torch.logical_or(pred_flat == 1, gt_flat == 1).sum(dim=1).float()

    # 处理分母为 0 的情况：如果并集为 0，IoU 设为 1（通常指背景全对）
    # 但在指代分割中，通常建议设为 0 或极小值 eps
    ious = intersection / (union + 1e-6)

    # 返回交集总和、并集总和以及每个样本的 iou
    return intersection.sum().item(), union.sum().item(), ious


# def visualize_all_scales_grid(original_img, gt_mask, pred_mask, g_list, iou_value, save_dir, raw):
#     """
#     动态可视化全尺度激活图。
#     g_list 可以是 [g3_c, g3_t, g2_c, g2_t, g1_t] 等任意长度。
#     布局：[Original Image] + [g_list 内容...] + [Pred vs GT]
#     """
#     # 1. 图像反标准化
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(original_img.device)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(original_img.device)
#     img_denorm = torch.clamp(original_img * std + mean, 0, 1)
#     img_np = img_denorm.cpu().numpy().transpose(1, 2, 0)
#
#     # 2. 归一化处理函数
#     def process_grid(grid):
#         if grid is None:  # 处理可能存在的 None 占位
#             return np.zeros((32, 32))
#         grid = grid.detach().cpu().squeeze()
#         if grid.dim() > 2: grid = grid[0]
#         g_min, g_max = grid.min(), grid.max()
#         return (grid - g_min) / (g_max - g_min + 1e-8)
#
#     # 3. 准备子图标签 (根据你当前 [g3_c, g3_t, g2_c, g2_t, g1_t] 的顺序)
#     # 如果列表长度变化，它会自动使用序号
#     default_names = ['C-S3', 'T-S3', 'C-S2', 'T-S2', 'T-S1', 'Refine-6']
#
#     # 4. 计算布局
#     num_grids = len(g_list)
#     total_cols = num_grids + 2
#     fig, axes = plt.subplots(1, total_cols, figsize=(5 * total_cols, 5))
#     if total_cols == 1: axes = [axes]  # 防止只有一个子图时 axes 不是列表
#
#     # --- 子图 0: 原图 ---
#     axes[0].imshow(img_np)
#     axes[0].set_title('Original Image', fontsize=15)
#
#     # --- 中间部分: 动态激活图 ---
#     for i in range(num_grids):
#         ax_idx = i + 1
#         grid_data = process_grid(g_list[i])
#
#         # 自动分配标题
#         title = default_names[i] if i < len(default_names) else f'Scale-{i}'
#
#         # 使用 jet 颜色映射，interpolation 使用 bilinear 使热力图平滑
#         im = axes[ax_idx].imshow(grid_data, cmap='jet', interpolation='bilinear')
#         axes[ax_idx].set_title(title, fontsize=15)
#         fig.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)
#
#     # --- 最后一个子图: Pred vs GT ---
#     gt_visual = (gt_mask.cpu().numpy() > 0).astype(np.float32)
#     if pred_mask.dim() == 3:  # 处理 [C, H, W]
#         pred_visual = (pred_mask.argmax(dim=0).cpu().numpy() > 0).astype(np.float32)
#     else:  # 处理 [H, W]
#         pred_visual = (pred_mask.cpu().numpy() > 0).astype(np.float32)
#
#     axes[-1].imshow(pred_visual, cmap='gray')
#     # 绘制 GT 红色轮廓
#     try:
#         if gt_visual.max() > 0:
#             axes[-1].contour(gt_visual, colors='red', linewidths=2.0, levels=[0.5])
#     except Exception as e:
#         pass
#
#     axes[-1].set_title(f'Pred vs GT\n(IoU: {iou_value:.3f})', fontsize=15)
#
#     # 5. 通用格式设置
#     for ax in axes:
#         ax.axis('off')
#
#     plt.tight_layout()
#
#     # 6. 保存
#     os.makedirs(save_dir, exist_ok=True)
#     safe_iou = f"{iou_value:.3f}".replace('.', '_')
#     filename = f"{raw}_{safe_iou}.png"
#     save_path = os.path.join(save_dir, filename)
#     plt.savefig(save_path, bbox_inches='tight', dpi=100)
#     plt.close()


def visualize_all_scales_grid(original_img, gt_mask, pred_mask, g_list, iou_value, save_dir, raw):
    """
    修改版：将原图叠加到各尺度激活图中，并保持间距适中且无标题。
    """
    # 1. 图像反标准化 (得到 HWC, RGB, [0,1])
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(original_img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(original_img.device)
    img_denorm = torch.clamp(original_img * std + mean, 0, 1)
    img_np = img_denorm.cpu().numpy().transpose(1, 2, 0)

    h, w, _ = img_np.shape

    # 2. 准备子图布局
    num_grids = len(g_list)
    total_cols = num_grids + 2  # 原图 + 叠加图... + 预测结果
    fig, axes = plt.subplots(1, total_cols, figsize=(4 * total_cols, 4))

    if total_cols == 1: axes = [axes]

    # --- 子图 0: 纯原图 ---
    axes[0].imshow(img_np)

    # --- 中间部分: 叠加后的激活图 ---
    for i in range(num_grids):
        ax_idx = i + 1
        grid = g_list[i]

        if grid is not None:
            # 处理 grid 数据
            grid = grid.detach().cpu().squeeze()
            if grid.dim() > 2: grid = grid[0]

            # 归一化到 [0, 1]
            g_min, g_max = grid.min(), grid.max()
            grid_norm = (grid - g_min) / (g_max - g_min + 1e-8)
            grid_np = grid_norm.numpy()

            # 将激活图 resize 到原图尺寸
            heatmap_resized = cv2.resize(grid_np, (w, h), interpolation=cv2.INTER_LINEAR)

            # 使用 jet 映射并将热力图应用到原图上
            # 步骤：显示原图 -> 显示热力图（设置 alpha 透明度）
            axes[ax_idx].imshow(img_np)
            axes[ax_idx].imshow(heatmap_resized, cmap='jet', alpha=0.5, interpolation='bilinear')
        else:
            axes[ax_idx].imshow(img_np)  # 如果没有数据则显示原图占位

    # --- 最后一个子图: Pred vs GT (同样建议叠加在原图上，看得更清楚) ---
    gt_visual = (gt_mask.cpu().numpy() > 0).astype(np.float32)
    if pred_mask.dim() == 3:
        pred_visual = (pred_mask.argmax(dim=0).cpu().numpy() > 0).astype(np.float32)
    else:
        pred_visual = (pred_mask.cpu().numpy() > 0).astype(np.float32)

    axes[-1].imshow(img_np)
    # 用半透明绿色表示预测区域
    mask_overlay = np.zeros_like(img_np)
    mask_overlay[pred_visual > 0] = [0, 1, 0]  # 绿色
    axes[-1].imshow(mask_overlay, alpha=0.3)

    # 绘制 GT 红色轮廓
    try:
        if gt_visual.max() > 0:
            axes[-1].contour(gt_visual, colors='red', linewidths=2.0, levels=[0.5])
    except:
        pass

    # 3. 格式清理
    for ax in axes:
        ax.axis('off')

    # 调整 wspace 控制子图之间的水平间距 (0.1 为适中)
    plt.subplots_adjust(wspace=0.1, hspace=0, left=0, right=1, bottom=0, top=1)

    # 4. 保存
    os.makedirs(save_dir, exist_ok=True)
    safe_iou = f"{iou_value:.3f}".replace('.', '_')
    filename = f"{raw}_{safe_iou}.png"
    save_path = os.path.join(save_dir, filename)

    plt.savefig(save_path, bbox_inches='tight', dpi=150, pad_inches=0.05)
    plt.close()


# def validate_epoch(args, val_loader, model, logger, mode='val', val_epoch=99):
#     logger.info(f'--- Starting {mode} Epoch {val_epoch} ---')
#     visualize = args.visulize == 1
#     device = torch.cuda.current_device()
#     tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer)
#
#     # 计数器初始化
#     batch_time = AverageMeter()
#     miou_meter = AverageMeter()
#     total_inter, total_union = 0.0, 0.0
#     prec_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
#     prec_meters = {t: AverageMeter() for t in prec_thresholds}
#
#     # 用于存储详细 IoU 的列表
#     detailed_results = []
#
#     model.eval()
#     end = time.time()
#     head_path = f'test_sample/epoch_{val_epoch}/'
#     if visualize:
#         os.makedirs(head_path, exist_ok=True)
#
#     torch.cuda.empty_cache()
#
#     for batch_idx, (imgs, seg_map, word_id, mask) in enumerate(val_loader):
#         imgs = imgs.to(device)
#         word_id = word_id.to(device)
#         seg_map = seg_map.to(device)
#         mask = mask.to(device)
#
#         with torch.no_grad():
#             num_descs = word_id.size(-1)
#             for j in range(num_descs):
#                 # 模型前向传播 (batch_size=1)
#                 mask_out, g = model(imgs, word_id[:, :, :, j], mask[:, :, :, j])
#                 batch_inter_sum, batch_union_sum, ious = compute_batch_iou(mask_out, seg_map)
#
#                 # 更新指标
#                 total_inter += batch_inter_sum
#                 total_union += batch_union_sum
#                 miou_meter.update(ious.mean().item(), imgs.size(0))
#                 for t in prec_thresholds:
#                     p_at_t = (ious > t).float().mean().item()
#                     prec_meters[t].update(p_at_t, imgs.size(0))
#
#                 # --- 核心记录逻辑 ---
#                 # 解码文字 query (k=0 因为 batch_size=1)
#                 text_query = word_ids_to_string(word_id[0, :, :, j], tokenizer)
#                 iou_val = ious[0].item()
#
#                 # 格式: batchIdx_queryIdx_textQuery_iou
#                 # 使用 "_" 作为分隔符，或者使用 "\t" 方便后续程序分割处理
#                 record = f"{batch_idx}_{j}_{text_query}_{iou_val:.6f}"
#                 detailed_results.append(record)
#
#         torch.cuda.empty_cache()
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if batch_idx % 100 == 0:  # 调小一点方便观察进度
#             logger.info(f'[{batch_idx}/{len(val_loader)}] mIoU {miou_meter.avg:.4f}')
#
#     # --- 保存为 TXT 文件 ---
#     output_txt = f"iou_details_epoch{val_epoch}.txt"
#     with open(output_txt, 'w', encoding='utf-8') as f:
#         for line in detailed_results:
#             f.write(line + '\n')
#     logger.info(f'>>> All image results saved to {output_txt}')
#
#     # 最终报告逻辑保持不变
#     final_miou = miou_meter.avg
#     final_oiou = total_inter / (total_union + 1e-6)
#     final_prec = {t: prec_meters[t].avg for t in prec_thresholds}
#
#     # ... (logger 打印逻辑) ...
#
#     return final_miou, final_oiou, final_prec

def visualize_comparison(original_img, gt_mask, pred_mask, iou_value, save_dir, filename):
    """
    只生成一张图：原图上叠加绿色预测掩码和红色GT轮廓。
    """
    # 1. 图像反标准化 (Device -> CPU, [C,H,W] -> [H,W,C], [0,1])
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(original_img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(original_img.device)
    img_denorm = torch.clamp(original_img * std + mean, 0, 1)
    img_np = img_denorm.cpu().numpy().transpose(1, 2, 0)

    h, w, _ = img_np.shape

    # 2. 准备 Mask 数据
    gt_visual = (gt_mask.cpu().numpy() > 0).astype(np.uint8)
    # 处理 pred_mask：如果是 (2, H, W) 取 argmax，如果是 (H, W) 直接阈值
    if pred_mask.dim() == 3:
        pred_visual = (pred_mask.argmax(dim=0).cpu().numpy() > 0).astype(np.uint8)
    else:
        pred_visual = (pred_mask.cpu().numpy() > 0).astype(np.uint8)

    # 3. 绘图
    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)

    # 叠加半透明绿色预测区域
    mask_overlay = np.zeros_like(img_np)
    mask_overlay[pred_visual > 0] = [0, 1, 0]  # 纯绿色
    plt.imshow(mask_overlay, alpha=0.35)  # 35%透明度

    # 绘制红色 GT 轮廓
    if gt_visual.max() > 0:
        # 使用 matplotlib 的 contour 绘制轮廓更平滑
        plt.contour(gt_visual, colors='red', linewidths=2.5, levels=[0.5])

    plt.axis('off')

    # 在图片左上角打上 IoU 数值（可选）
    # plt.text(10, 30, f'IoU: {iou_value:.4f}', color='white',
    #          fontsize=15, fontweight='bold', backgroundcolor='black')

    # 4. 保存
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename}.png")

    # 使用 savefig 保存，去掉白边
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()


# def validate_epoch(args, val_loader, model, logger, mode='val', val_epoch=99, target_ids=[114,40,14,18,109,127,1,13,66,157,69,15,4]):
#     """
#     target_ids: 指定要可视化的索引列表，如 [14, 40, 114, 180]
#     """
#     logger.info(f'--- Starting {mode} Epoch {val_epoch} ---')
#     device = torch.cuda.current_device()
#     tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer)
#
#     model.eval()
#
#     # 结果保存路径
#     save_path = f'visual_results/epoch_{val_epoch}_comparison/'
#
#     # 确保 target_ids 是集合，查找更快
#     target_set = set(target_ids) if target_ids else set()
#
#     for batch_idx, (imgs, seg_map, word_id, mask) in enumerate(val_loader):
#         # 只要当前 batch_idx 不在我们要找的列表里，直接跳过推理阶段（节省时间）
#         # 如果你还需要计算 mIoU，则不能跳过推理，只需跳过可视化逻辑
#         # if target_ids is not None and batch_idx not in target_set:
#         #     continue
#
#         imgs = imgs.to(device)
#         word_id = word_id.to(device)
#         seg_map = seg_map.to(device)
#         mask = mask.to(device)
#
#         with torch.no_grad():
#             num_descs = word_id.size(-1)
#             for j in range(num_descs):
#                 # 模型推理 (Batch Size = 1)
#                 mask_out, _ = model(imgs, word_id[:, :, :, j], mask[:, :, :, j])
#                 _, _, ious = compute_batch_iou(mask_out, seg_map)
#
#                 iou_val = ious[0].item()
#                 text_query = word_ids_to_string(word_id[0, :, :, j], tokenizer)
#
#                 # 生成文件名：ID_方位词关键部分_IoU
#                 clean_query = "".join([c for c in text_query if c.isalnum() or c == ' ']).replace(' ', '_')[:30]
#                 file_name = f"ID_{batch_idx}_Q{j}_{clean_query}"
#
#                 # 调用精简版可视化
#                 visualize_comparison(
#                     imgs[0], seg_map[0], mask_out[0],
#                     iou_val, save_path, file_name
#                 )
#
#                 logger.info(f"Visualized target ID {batch_idx}, Query: {text_query}, IoU: {iou_val:.4f}")
#
#     return None  # 如果只是为了可视化，可以不返回指标

def validate_epoch(args, val_loader, model, logger, mode='val', val_epoch=99):
    logger.info(f'--- Starting {mode} Epoch {val_epoch} ---')
    visualize = args.visulize == 1
    device = torch.cuda.current_device()
    tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer)
    # 计数器初始化
    batch_time = AverageMeter()
    miou_meter = AverageMeter()

    # oIoU 累加器
    total_inter = 0.0
    total_union = 0.0

    # Precision@X 阈值定义
    prec_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    prec_meters = {t: AverageMeter() for t in prec_thresholds}

    model.eval()
    end = time.time()
    save_good = 5
    save_bad = 5
    bad_cnt = 0
    good_cnt = 0
    head_path = f'test_sample/epoch_{val_epoch}/'

    if visualize:
        os.makedirs(head_path, exist_ok=True)

    # 清空显存开始验证
    torch.cuda.empty_cache()

    for batch_idx, (imgs, seg_map, word_id, mask) in enumerate(val_loader):
        # 搬运数据
        imgs = imgs.to(device)
        word_id = word_id.to(device)
        seg_map = seg_map.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            num_descs = word_id.size(-1)
            for j in range(num_descs):
                # 模型前向传播
                mask_out, g = model(imgs, word_id[:, :, :, j], mask[:, :, :, j])
                batch_inter_sum, batch_union_sum, ious = compute_batch_iou(mask_out, seg_map)
                total_inter += batch_inter_sum
                total_union += batch_union_sum
                miou_meter.update(ious.mean().item(), imgs.size(0))
                # 更新 Precision@X
                for t in prec_thresholds:
                    p_at_t = (ious > t).float().mean().item()
                    prec_meters[t].update(p_at_t, imgs.size(0))
                if visualize and good_cnt + bad_cnt < 150:
                    for k in range(imgs.size(0)):
                        if ious[k] < 0.3 and bad_cnt < save_bad:
                            bad_cnt += 1
                            text_query = word_ids_to_string(word_id[k, :, :, j], tokenizer)
                            visualize_all_scales_grid(
                                imgs[k], seg_map[k], mask_out[k],
                                g, ious[k], head_path, text_query
                            )

                        if ious[k] > 0.8 and good_cnt < save_good:
                            text_query = word_ids_to_string(word_id[k, :, :, j], tokenizer)
                            good_cnt += 1
                            # 优化：只保存激活图，不保存中间变量
                            visualize_all_scales_grid(
                                imgs[k], seg_map[k], mask_out[k],
                                g, ious[k], head_path, text_query
                            )

        # 每个 batch 后清理显存
        torch.cuda.empty_cache()
        # 耗时统计
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 1000 == 0:
            logger.info(f'[{batch_idx}/{len(val_loader)}] '
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        f'mIoU {miou_meter.val:.4f} ({miou_meter.avg:.4f})')

    # 最终清理
    torch.cuda.empty_cache()
    # 计算最终整体指标
    final_miou = miou_meter.avg
    final_oiou = total_inter / (total_union + 1e-6)
    final_prec = {t: prec_meters[t].avg for t in prec_thresholds}

    # 打印最终报告
    logger.info("=" * 30)
    logger.info(f"FINAL {mode.upper()} RESULTS:")
    logger.info(f"mIoU: {final_miou:.4f}")
    logger.info(f"oIoU: {final_oiou:.4f}")
    for t in prec_thresholds:
        logger.info(f"Pr@{int(t * 100)}: {final_prec[t]:.4f}")
    logger.info("=" * 30)

    return final_miou, final_oiou, final_prec


def train_epoch(rank, args, train_loader, model, optimizer, epoch, scaler, logger, lr_scheduler):
    # DDP 模式下通常在外部设置 sampler.set_epoch(epoch)
    if rank == 0:
        logger.info(f'--- Training Epoch {epoch} ---')

    batch_time = AverageMeter()
    losses = AverageMeter()
    ce_losses = AverageMeter()
    dice_losses = AverageMeter()
    iou_meter = AverageMeter()  # 改名为 iou_meter 更直观

    model.train()
    end = time.time()

    # 获取分布式训练的设备
    device = torch.device(f'cuda:{rank}')

    for batch_idx, (imgs, seg_map, word_id, word_mask) in enumerate(train_loader):
        # 1. 数据搬运 (移除过时的 Variable 包装)
        imgs = imgs.to(device, non_blocking=True)
        word_id = word_id.to(device, non_blocking=True)
        word_mask = word_mask.to(device, non_blocking=True)
        seg_map = seg_map.to(device, non_blocking=True)

        # 2. 前向传播 (混合精度)
        with autocast():
            # DDPA 模型输出
            mask_out = model(imgs, word_id, word_mask)

            # 计算损失 (根据你的 criterion 结构适配)
            # 假设 criterion 返回 total_loss, ce_loss, dice_loss
            loss, ce_loss, dice_loss = criterion(mask_out, seg_map, epoch)

            # 3. 计算训练 IoU (使用新版函数，解包三个值)
            # 训练阶段主要关注 mIoU 作为趋势参考
            _, _, ious = compute_batch_iou(mask_out, seg_map)
            current_miou = ious.mean().item()

        # 4. 反向传播与优化
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # 学习率调整 (如果是每 step 调整)
        lr_scheduler.step()

        # 5. 更新统计量
        losses.update(loss.item(), imgs.size(0))
        ce_losses.update(ce_loss.item() if torch.is_tensor(ce_loss) else ce_loss, imgs.size(0))
        dice_losses.update(dice_loss.item() if torch.is_tensor(dice_loss) else dice_loss, imgs.size(0))
        iou_meter.update(current_miou, imgs.size(0))

        # 6. 时间统计 (移除同步以保持性能)
        batch_time.update(time.time() - end)
        end = time.time()

        # 7. 日志打印 (仅在主进程 rank 0 打印)
        if rank == 0 and batch_idx % args.print_freq == 0:
            curr_lr = optimizer.param_groups[0]['lr']
            print_str = (
                'Epoch: [{0}][{1}/{2}]  '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                'CE {ce:.4f}  Dice {dice:.4f}  '
                'IoU {iou.val:.4f} ({iou.avg:.4f})  '
                'LR {lr:.8f}'
                .format(epoch, batch_idx, len(train_loader),
                        batch_time=batch_time, loss=losses,
                        ce=ce_losses.val, dice=dice_losses.val,
                        iou=iou_meter, lr=curr_lr))
            logger.info(print_str)
    # 周期结束，仅在主进程输出总结
    if rank == 0:
        logger.info(f"Epoch {epoch} Train Summary: Loss {losses.avg:.4f}, IoU {iou_meter.avg:.4f}")

    return iou_meter.avg
