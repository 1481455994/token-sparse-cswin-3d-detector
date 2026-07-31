"""
无锚框3D目标检测头 (Anchor-Free 3D Detection Head)
基于YOLOv8官方无锚框核心机制设计
适用于96×96×96输入的多尺度特征金字塔检测

核心修改：
1. 解耦头结构：分类分支、回归分支、置信度分支完全分离
2. YOLOv8标准回归：直接回归6个边界距离 (z1, y1, x1, z2, y2, x2)
3. Task-Aligned Assigner：每个GT仅分配1~9个正样本
4. 合理的损失归一化：按batch_size归一化，避免loss被稀释
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math
import numpy as np

# 尝试导入DFL Loss
try:
    from .losses.dfl_loss import DFLLoss, DFLCIoULoss
    HAS_DFL = True
except ImportError:
    HAS_DFL = False
    print("[WARNING] DFL Loss not available, falling back to CIoU only")

# 尝试导入Focal Loss
try:
    from .losses.detection_losses import FocalLoss
    HAS_FOCAL = True
except ImportError:
    HAS_FOCAL = False
    print("[WARNING] Focal Loss not available, falling back to BCE")


class AnchorFreeHead3D(nn.Module):
    """
    YOLOv8风格无锚框3D检测头

    采用解耦头设计：分类分支、回归分支完全分离

    输出格式（每个体素）:
        - 分类分支: num_classes (目标置信度分数，与YOLOv8一致)
        - 回归分支: 6 (z1, y1, x1, z2, y2, x2 边界距离)

    关键修改（方案A）：
    - 删除独立的 conf_pred 分支
    - 分类分数直接作为检测分数（与YOLOv8一致）
    - BCE loss 对所有位置计算
    """

    def __init__(
        self,
        in_channels_list: List[int],
        feat_channels: int = 256,
        num_classes: int = 1,
        num_levels: int = 3,
        use_gn: bool = True,
        use_dfl: bool = False,
        dfl_bins: int = 18,
        dfl_range_low: float = 0.0,
        dfl_range_high: float = 48.0,
    ):
        """
        Args:
            in_channels_list: 各尺度特征图通道数 [192, 128, 96] 对应 [Level0(12³), Level1(24³), Level2(48³)]
            feat_channels: 检测头内部特征通道数
            num_classes: 类别数（默认1：结节/非结节二分类）
            num_levels: 特征金字塔层数
            use_gn: 是否使用GroupNorm
            use_dfl: 是否使用DFL Loss
            dfl_bins: DFL的bins数量 (默认18: 2-10步长1为8bins, 10-30步长2为10bins)
        """
        super().__init__()
        self.in_channels_list = in_channels_list
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        self.num_levels = num_levels
        self.use_dfl = use_dfl and HAS_DFL
        self.dfl_bins = dfl_bins
        self.dfl_range_low = dfl_range_low
        self.dfl_range_high = dfl_range_high

        # 为每层特征图创建解耦检测头（分类、回归二分支）
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            # ========== 分类分支 (Classification) ==========
            # 输出: num_classes 通道，分类分数直接作为检测置信度
            cls_adapter = nn.Conv3d(in_channels, feat_channels, kernel_size=3, padding=1, bias=False)
            cls_norm = nn.GroupNorm(32, feat_channels) if use_gn else nn.BatchNorm3d(feat_channels)
            cls_act = nn.SiLU(inplace=True)
            cls_conv2 = nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False)
            cls_norm2 = nn.GroupNorm(32, feat_channels) if use_gn else nn.BatchNorm3d(feat_channels)
            cls_act2 = nn.SiLU(inplace=True)
            cls_out = nn.Conv3d(feat_channels, num_classes, kernel_size=1)

            cls_conv = nn.Sequential(
                cls_adapter, cls_norm, cls_act,
                cls_conv2, cls_norm2, cls_act2,
                cls_out
            )

            # ========== 回归分支 (Regression) ==========
            # 输出: 6通道 (z1, y1, x1, z2, y2, x2 相对于特征点中心的距离)
            # 如果使用DFL，则输出 6 * dfl_bins 通道
            reg_out_channels = 6 * dfl_bins if self.use_dfl else 6
            reg_adapter = nn.Conv3d(in_channels, feat_channels, kernel_size=3, padding=1, bias=False)
            reg_norm = nn.GroupNorm(32, feat_channels) if use_gn else nn.BatchNorm3d(feat_channels)
            reg_act = nn.SiLU(inplace=True)
            reg_conv2 = nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False)
            reg_norm2 = nn.GroupNorm(32, feat_channels) if use_gn else nn.BatchNorm3d(feat_channels)
            reg_act2 = nn.SiLU(inplace=True)
            reg_out = nn.Conv3d(feat_channels, reg_out_channels, kernel_size=1)

            reg_conv = nn.Sequential(
                reg_adapter, reg_norm, reg_act,
                reg_conv2, reg_norm2, reg_act2,
                reg_out
            )

            self.cls_convs.append(cls_conv)
            self.reg_convs.append(reg_conv)

        # 初始化DFL损失模块
        if self.use_dfl:
            self.dfl_loss = DFLLoss(
                range_low=dfl_range_low,
                range_high=dfl_range_high,
                fine_bins=8,      # 2-10, 步长1, 8个bins
                coarse_bins=10,   # 10-30, 步长2, 10个bins
            )
            print(
                f"[AnchorFreeHead3D] DFL enabled with {dfl_bins} bins, "
                f"range=[{dfl_range_low}, {dfl_range_high}]"
            )

    def forward(self, features: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """
        前向传播

        Args:
            features: 三尺度特征金字塔 [Level0(12³,192ch), Level1(24³,128ch), Level2(48³,96ch)]

        Returns:
            cls_scores: 每层分类结果 [B, num_classes, D, H, W]（直接作为检测置信度）
            reg_preds: 每层回归结果 [B, 6, D, H, W] (z1, y1, x1, z2, y2, x2 距离)
        """
        cls_scores = []
        reg_preds = []

        for level_idx in range(len(features)):
            feat = features[level_idx]

            # 分类预测（直接作为检测置信度）
            cls_score = self.cls_convs[level_idx](feat)

            # 回归预测
            reg_pred = self.reg_convs[level_idx](feat)

            cls_scores.append(cls_score)
            reg_preds.append(reg_pred)

        return {
            'cls_scores': cls_scores,
            'reg_preds': reg_preds,
        }


class AnchorFreeLoss3D(nn.Module):
    """
    YOLOv8风格无锚框3D检测损失函数

    损失组成：
    - 分类损失：BCEWithLogitsLoss (所有样本，包括正负样本)
    - 框回归损失：3D DIoU (仅正样本)

    关键改进（方案A）：
    1. 删除独立的置信度分支，分类分数直接作为检测置信度
    2. BCE loss 对所有位置计算（不只是正样本）
    3. 参考YOLOv8: target_scores由assigner生成，正样本≈1，负样本≈0
    """

    def __init__(
        self,
        num_classes: int = 1,
        cls_weight: float = 1.0,
        bbox_weight: float = 2.0,
        loss_type: str = 'diou',
        neg_iou_thr: float = 0.5,  # 负样本IoU阈值
        max_pos_per_gt: int = 3,    # 每个GT最多分配的正样本数量
        neg_sample_random_ratio: float = 1.0,   # 随机采样负样本比例 (0-1)
        neg_sample_hard_ratio: float = 0.01,    # 困难负样本挖掘比例 (0-1)
        use_dfl: bool = False,
        dfl_weight: float = 0.25,
        ciou_weight: float = 0.75,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        assignment_mode: str = "classic",
        assign_use_pred_iou: bool = True,
        assign_one_level: bool = False,
        assignment_quality: str = "distance",
        assign_alpha: float = 1.0,
        assign_beta: float = 6.0,
        assign_expand_ratio: float = 1.5,
        assign_ignore_expand_ratio: float = 1.1,
        assign_soft_target_scores: bool = False,
        dfl_range_low: float = 0.0,
        dfl_range_high: float = 48.0,
        loss_normalization: str = "legacy",
        neg_cls_weight: float = 1.0,
        pos_cls_weight: float = 1.0,
        bbox_loss_norm: str = "num_pos",
        use_hard_negative_weight: bool = False,
        hard_negative_cls_weight: float = 1.5,
        hard_negative_score_threshold: float = 0.3,
    ):
        """
        Args:
            num_classes: 类别数
            cls_weight: 分类损失权重
            bbox_weight: 框回归损失权重
            loss_type: 损失类型 'diou' 或 'ciou'
            neg_iou_thr: 负样本IoU阈值 (与所有GT的IoU低于此值才算负样本)
            max_pos_per_gt: 每个GT最多分配的正样本数量
            neg_sample_random_ratio: 随机采样负样本的比例 (0-1)，1表示全部保留
            neg_sample_hard_ratio: 困难负样本挖掘的比例 (0-1)，1%表示保留损失最大的1%
            use_dfl: 是否使用DFL Loss
            dfl_weight: DFL损失权重 (仅当use_dfl=True时有效)
            ciou_weight: CIoU损失权重 (仅当use_dfl=True时有效)
            focal_alpha: Focal Loss的正样本权重
            focal_gamma: Focal Loss的困难样本聚焦因子
        """
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.loss_type = str(loss_type).lower()
        self.neg_iou_thr = neg_iou_thr
        self.max_pos_per_gt = max_pos_per_gt
        self.neg_sample_random_ratio = neg_sample_random_ratio
        self.neg_sample_hard_ratio = neg_sample_hard_ratio
        self.use_dfl = use_dfl and HAS_DFL
        self.dfl_weight = dfl_weight
        self.ciou_weight = ciou_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.assignment_mode = str(assignment_mode).lower()
        self.assign_use_pred_iou = assign_use_pred_iou
        self.assign_one_level = assign_one_level
        self.assignment_quality = str(assignment_quality).lower()
        self.assign_alpha = float(assign_alpha)
        self.assign_beta = float(assign_beta)
        self.assign_expand_ratio = float(assign_expand_ratio)
        self.assign_ignore_expand_ratio = float(assign_ignore_expand_ratio)
        self.assign_soft_target_scores = bool(assign_soft_target_scores)
        self.loss_normalization = str(loss_normalization).lower()
        self.neg_cls_weight = float(neg_cls_weight)
        self.pos_cls_weight = float(pos_cls_weight)
        self.bbox_loss_norm = str(bbox_loss_norm).lower()
        self.use_hard_negative_weight = bool(use_hard_negative_weight)
        self.hard_negative_cls_weight = float(hard_negative_cls_weight)
        self.hard_negative_score_threshold = float(hard_negative_score_threshold)
        if self.loss_type not in {"diou", "ciou"}:
            raise ValueError(f"Unsupported loss_type={loss_type!r}. Expected 'diou' or 'ciou'.")
        if self.bbox_loss_norm not in {"num_pos", "batch_size"}:
            raise ValueError(
                f"Unsupported bbox_loss_norm={bbox_loss_norm!r}. Expected 'num_pos' or 'batch_size'."
            )
        if self.assignment_mode not in {"classic", "global", "multi_level", "one_level"}:
            raise ValueError(
                f"Unsupported assignment_mode={assignment_mode!r}. "
                "Expected 'classic', 'global', 'multi_level', or 'one_level'."
            )
        if self.assignment_quality not in {"distance", "pred_iou", "center_topk"}:
            raise ValueError(
                f"Unsupported assignment_quality={assignment_quality!r}. "
                "Expected 'distance', 'pred_iou', or 'center_topk'."
            )

        # 使用Focal Loss用于分类（优先），BCE作为备选
        if HAS_FOCAL:
            self.cls_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='none')
            print(f"[AnchorFreeLoss3D] Focal Loss enabled: alpha={focal_alpha}, gamma={focal_gamma}")
        else:
            self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            print("[AnchorFreeLoss3D] Focal Loss not available, using BCEWithLogitsLoss")

        # 初始化DFL损失模块
        if self.use_dfl:
            self.dfl_loss = DFLLoss(
                range_low=dfl_range_low,
                range_high=dfl_range_high,
                fine_bins=8,      # 2-10, 步长1, 8个bins
                coarse_bins=10,   # 10-30, 步长2, 10个bins
            )
            print(
                f"[AnchorFreeLoss3D] DFL enabled: dfl_weight={dfl_weight}, "
                f"ciou_weight={ciou_weight}, range=[{dfl_range_low}, {dfl_range_high}]"
            )

    def forward(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        targets: List[Dict],
        feature_sizes: List[List[int]],
        image_shape: Tuple[int, int, int],
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失

        Args:
            predictions: 检测头输出 {'cls_scores': [...], 'reg_preds': [...]}
            targets: GT标注列表 [{'boxes': [M,6], 'labels': [M]}, ...]
            feature_sizes: 特征图尺寸列表
            image_shape: 原图尺寸 (D, H, W)
            debug: 是否打印调试信息

        Returns:
            损失字典
        """
        cls_scores = predictions['cls_scores']
        reg_preds = predictions['reg_preds']

        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]

        # 累积损失
        total_cls_loss = torch.tensor(0.0, device=device)
        total_cls_loss_pos = torch.tensor(0.0, device=device)
        total_cls_loss_neg = torch.tensor(0.0, device=device)
        total_bbox_loss = torch.tensor(0.0, device=device)
        total_dfl_loss = torch.tensor(0.0, device=device)

        total_pos_samples = 0
        total_neg_samples = 0
        all_pos_cls_losses = []
        all_neg_cls_losses = []
        all_neg_scores = []
        all_neg_is_hard_negative = []
        all_bbox_losses = []
        all_dfl_losses = []

        for b in range(batch_size):
            gt_boxes = targets[b]['boxes']
            gt_labels = targets[b]['labels'] if 'labels' in targets[b] else torch.zeros(
                gt_boxes.shape[0], dtype=torch.long, device=device
            )
            ignored_boxes = targets[b].get(
                'ignored_boxes',
                torch.zeros((0, 6), dtype=gt_boxes.dtype, device=device),
            )
            image_path = targets[b].get('image_path', 'unknown')
            is_hard_negative = targets[b].get('is_hard_negative', False)
            if isinstance(is_hard_negative, torch.Tensor):
                is_hard_negative = bool(is_hard_negative.item())
            else:
                is_hard_negative = bool(is_hard_negative)

            if debug:
                print(f"\n[DEBUG Batch{b}] Image: {image_path}")
                print(f"[DEBUG Batch{b}] GT boxes count: {gt_boxes.shape[0]}")

            level_data = []
            for level_idx in range(len(cls_scores)):
                cls_score = cls_scores[level_idx][b]   # [C, D, H, W]
                reg_pred = reg_preds[level_idx][b]     # [6, D, H, W] 或 [6*18, D, H, W]

                feat_size = feature_sizes[level_idx]

                # 展平: [C, D, H, W] -> [N, C]
                C, D, H, W = cls_score.shape
                cls_score_flat = cls_score.permute(1, 2, 3, 0).reshape(-1, C)  # [N, C]

                # 回归预测展平（支持DFL的分布输出）
                if self.use_dfl:
                    # [6*18, D, H, W] -> [D, H, W, 6, 18] -> [N, 6, 18]
                    reg_pred_flat = reg_pred.permute(1, 2, 3, 0).reshape(D*H*W, 6, self.dfl_loss.num_bins)
                else:
                    # [6, D, H, W] -> [N, 6]
                    reg_pred_flat = reg_pred.permute(1, 2, 3, 0).reshape(-1, 6)

                # 生成该层的中心点网格
                center_points = self._generate_grid(feat_size, image_shape, device)  # [N, 3]

                level_data.append({
                    'feat_size': feat_size,
                    'cls_score_flat': cls_score_flat,
                    'reg_pred_flat': reg_pred_flat,
                    'center_points': center_points,
                })

            level_assignments = self._task_aligned_assign_global(
                [item['center_points'] for item in level_data],
                [item['cls_score_flat'] for item in level_data],
                [item['reg_pred_flat'] for item in level_data],
                gt_boxes,
                gt_labels,
                ignored_boxes,
                image_shape,
                debug=debug,
                image_path=image_path,
                max_pos_per_gt=self.max_pos_per_gt,
            )

            for level_idx, item in enumerate(level_data):
                feat_size = item['feat_size']
                cls_score_flat = item['cls_score_flat']
                reg_pred_flat = item['reg_pred_flat']
                center_points = item['center_points']
                assignment = level_assignments[level_idx]
                pos_indices = assignment['pos_indices']
                neg_indices = assignment['neg_indices']
                target_scores = assignment['target_scores']
                gt_boxes_for_loss = assignment['gt_boxes_for_loss']

                n_pos = len(pos_indices)
                n_neg = len(neg_indices)
                total_pos_samples += n_pos
                total_neg_samples += n_neg

                # 每层统计（debug模式打印）
                if debug:
                    print(f"  [DEBUG Level{level_idx}] feat_size={feat_size}, N={len(center_points)}, "
                          f"n_pos={n_pos}, n_neg_before_sample={len(center_points) - n_pos}, n_neg_after_sample={n_neg}")

                # ========== 该 level 的分类损失 ==========
                target_scores_detached = target_scores.detach()
                level_cls_loss_pos = torch.tensor(0.0, device=device)
                level_cls_loss_neg = torch.tensor(0.0, device=device)

                if n_pos > 0 and n_neg > 0:
                    pos_cls_loss = self.cls_loss_fn(
                        cls_score_flat[pos_indices].squeeze(-1),
                        target_scores_detached[pos_indices]
                    )
                    neg_cls_loss = self.cls_loss_fn(
                        cls_score_flat[neg_indices].squeeze(-1),
                        target_scores_detached[neg_indices]
                    )
                    all_pos_cls_losses.append(pos_cls_loss.reshape(-1))
                    all_neg_cls_losses.append(neg_cls_loss.reshape(-1))
                    all_neg_scores.append(torch.sigmoid(cls_score_flat[neg_indices].squeeze(-1)).detach().reshape(-1))
                    all_neg_is_hard_negative.append(
                        torch.full((n_neg,), is_hard_negative, dtype=torch.bool, device=device)
                    )
                    level_cls_loss_pos = pos_cls_loss.sum()
                    level_cls_loss_neg = neg_cls_loss.sum()
                    level_cls_loss = level_cls_loss_pos + level_cls_loss_neg
                elif n_pos > 0:
                    pos_cls_loss = self.cls_loss_fn(
                        cls_score_flat[pos_indices].squeeze(-1),
                        target_scores_detached[pos_indices]
                    )
                    all_pos_cls_losses.append(pos_cls_loss.reshape(-1))
                    level_cls_loss_pos = pos_cls_loss.sum()
                    level_cls_loss = level_cls_loss_pos
                elif n_neg > 0:
                    neg_cls_loss = self.cls_loss_fn(
                        cls_score_flat[neg_indices].squeeze(-1),
                        target_scores_detached[neg_indices]
                    )
                    all_neg_cls_losses.append(neg_cls_loss.reshape(-1))
                    all_neg_scores.append(torch.sigmoid(cls_score_flat[neg_indices].squeeze(-1)).detach().reshape(-1))
                    all_neg_is_hard_negative.append(
                        torch.full((n_neg,), is_hard_negative, dtype=torch.bool, device=device)
                    )
                    level_cls_loss_neg = neg_cls_loss.sum()
                    level_cls_loss = level_cls_loss_neg
                else:
                    level_cls_loss = cls_score_flat.sum() * 0.0

                # ========== 该 level 的框回归损失 (仅正样本) ==========
                level_bbox_loss = torch.tensor(0.0, device=device)
                level_dfl_loss = torch.tensor(0.0, device=device)

                if n_pos > 0:
                    pos_reg = reg_pred_flat[pos_indices]
                    pos_center = center_points[pos_indices]

                    if self.use_dfl:
                        pos_reg_decoded = self.dfl_loss.decode_dist(pos_reg)
                        pred_boxes = self._decode_boxes_yolov8(pos_center, pos_reg_decoded, image_shape)
                        gt_boxes_tensor = torch.stack(gt_boxes_for_loss)
                        gt_boxes_corner = self._center_size_to_corners(gt_boxes_tensor)
                        dfl_loss, ciou_loss = self._compute_dfl_ciou_loss(
                            pos_center, pos_reg, gt_boxes_corner, image_shape
                        )
                        level_dfl_loss = dfl_loss
                        level_bbox_loss = ciou_loss.sum()
                        all_dfl_losses.append((dfl_loss * n_pos).reshape(-1))
                        all_bbox_losses.append(ciou_loss.reshape(-1))
                    else:
                        pred_boxes = self._decode_boxes_yolov8(pos_center, pos_reg, image_shape)
                        gt_boxes_tensor = torch.stack(gt_boxes_for_loss)
                        gt_boxes_corner = self._center_size_to_corners(gt_boxes_tensor)
                        bbox_loss = self._compute_bbox_loss(pred_boxes, gt_boxes_corner)
                        level_bbox_loss = bbox_loss.sum()
                        all_bbox_losses.append(bbox_loss.reshape(-1))
                else:
                    if self.use_dfl:
                        level_dfl_loss = cls_score_flat.sum() * 0.0
                    else:
                        level_bbox_loss = (cls_score_flat.sum() * 0.0).sum()

                # 累积该 level 的损失
                total_cls_loss = total_cls_loss + level_cls_loss
                total_cls_loss_pos = total_cls_loss_pos + level_cls_loss_pos
                total_cls_loss_neg = total_cls_loss_neg + level_cls_loss_neg
                total_bbox_loss = total_bbox_loss + level_bbox_loss
                if self.use_dfl:
                    total_dfl_loss = total_dfl_loss + level_dfl_loss

        # ========== 归一化：按batch_size归一化 ==========
        total_cls_loss = total_cls_loss / batch_size
        total_cls_loss_pos = total_cls_loss_pos / batch_size
        total_cls_loss_neg = total_cls_loss_neg / batch_size
        total_bbox_loss = total_bbox_loss / batch_size
        if self.use_dfl:
            total_dfl_loss = total_dfl_loss / batch_size

        # 总损失
        if self.use_dfl:
            # DFL + CIoU 组合损失
            total_loss = (
                self.cls_weight * total_cls_loss +
                total_dfl_loss * self.dfl_weight +  # DFL权重在_compute_dfl_ciou_loss中已应用
                total_bbox_loss * self.ciou_weight   # CIoU权重在_compute_dfl_ciou_loss中已应用
            )
        else:
            total_loss = (
                self.cls_weight * total_cls_loss +
                self.bbox_weight * total_bbox_loss
        )

        cls_loss_pos_norm = total_cls_loss * 0.0
        cls_loss_neg_norm = total_cls_loss * 0.0
        bbox_loss_norm = total_cls_loss * 0.0
        neg_loss_count = 0
        hard_negative_neg_count = 0

        if self.loss_normalization != "legacy":
            zero_loss = total_cls_loss * 0.0
            loss_norm = max(int(total_pos_samples), 1)

            if all_pos_cls_losses:
                cls_loss_pos_norm = torch.cat(all_pos_cls_losses).sum() / loss_norm

            if all_bbox_losses:
                bbox_losses = torch.cat(all_bbox_losses)
                bbox_norm = loss_norm if self.bbox_loss_norm == "num_pos" else batch_size
                bbox_loss_norm = bbox_losses.sum() / bbox_norm

            if self.use_dfl:
                total_dfl_loss = torch.cat(all_dfl_losses).sum() / loss_norm if all_dfl_losses else zero_loss

            if all_neg_cls_losses:
                neg_losses = torch.cat(all_neg_cls_losses)
                neg_scores = torch.cat(all_neg_scores) if all_neg_scores else torch.empty(0, device=device)
                neg_is_hard_negative = (
                    torch.cat(all_neg_is_hard_negative)
                    if all_neg_is_hard_negative
                    else torch.zeros_like(neg_losses, dtype=torch.bool)
                )
                hard_negative_neg_count = int(neg_is_hard_negative.sum().item())

                if self.use_hard_negative_weight and hard_negative_neg_count > 0:
                    high_score_mask = neg_scores >= self.hard_negative_score_threshold
                    hard_weight_mask = neg_is_hard_negative & high_score_mask
                    neg_losses = torch.where(
                        hard_weight_mask,
                        neg_losses * self.hard_negative_cls_weight,
                        neg_losses,
                    )

                # Keep negative classification pressure independent of num_pos.
                # This also preserves full strength on pure-negative patches.
                cls_loss_neg_norm = neg_losses.sum()
                neg_loss_count = int(neg_losses.numel())

            total_cls_loss_pos = cls_loss_pos_norm
            total_cls_loss_neg = cls_loss_neg_norm
            total_bbox_loss = bbox_loss_norm
            total_cls_loss = (
                self.pos_cls_weight * cls_loss_pos_norm +
                self.neg_cls_weight * cls_loss_neg_norm
            )

            if self.use_dfl:
                total_loss = (
                    self.cls_weight * total_cls_loss +
                    self.dfl_weight * total_dfl_loss +
                    self.ciou_weight * total_bbox_loss
                )
            else:
                total_loss = (
                    self.cls_weight * total_cls_loss +
                    self.bbox_weight * total_bbox_loss
                )

        result = {
            'loss': total_loss,
            'cls_loss': total_cls_loss,
            'cls_loss_pos': total_cls_loss_pos,
            'cls_loss_neg': total_cls_loss_neg,
            'bbox_loss': total_bbox_loss,
            'num_pos': total_pos_samples,
            'num_neg': total_neg_samples,
            'cls_loss_pos_norm': cls_loss_pos_norm,
            'cls_loss_neg_norm': cls_loss_neg_norm,
            'bbox_loss_norm': bbox_loss_norm,
            'neg_loss_count': neg_loss_count,
            'hard_negative_neg_count': hard_negative_neg_count,
        }
        if self.use_dfl:
            result['dfl_loss'] = total_dfl_loss

        return result

    def _generate_grid(
        self,
        feat_size: List[int],
        image_shape: Tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        生成特征图每个位置对应的原图中心点坐标
        """
        D, H, W = feat_size
        img_D, img_H, img_W = image_shape

        stride_z = img_D / D
        stride_y = img_H / H
        stride_x = img_W / W

        z_coords = (torch.arange(D, dtype=torch.float32, device=device) + 0.5) * stride_z
        y_coords = (torch.arange(H, dtype=torch.float32, device=device) + 0.5) * stride_y
        x_coords = (torch.arange(W, dtype=torch.float32, device=device) + 0.5) * stride_x

        grid_z, grid_y, grid_x = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        center_points = torch.stack([
            grid_z.flatten(),
            grid_y.flatten(),
            grid_x.flatten()
        ], dim=1)

        return center_points

    def _task_aligned_assign(
        self,
        center_points: torch.Tensor,
        cls_score_flat: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        feat_size: List[int],
        image_shape: Tuple[int, int, int],
        debug: bool = False,
        image_path: str = 'unknown',
        max_pos_per_gt: int = 3,  # 对应 YOLOv8 的 topk
        level_idx: int = 0,
    ) -> Tuple[List[int], List[int], torch.Tensor, Dict, List]:
        """
        【YOLOv8 动态分配版】Task-Aligned Assigner (TAA)
        
        核心逻辑：
        1. 计算当前层所有点与所有 GT 的 IoU。
        2. 计算对齐度 Metric = Classification^alpha * IoU^beta。
        3. 为每个 GT 选择 Metric 最高的 Top-K 个点。
        4. 动态去重：如果一个点被分配给多个 GT，只保留 Metric 最高的那个。
        """
        N = center_points.shape[0] # 该层特征点总数
        M = gt_boxes.shape[0] if gt_boxes.numel() > 0 else 0

        # 初始化所有位置的目标分数为0
        target_scores = torch.zeros(N, device=center_points.device)
        
        # 用于存储最终分配结果
        pos_to_gt_mapping = [] # List[Tuple[pos_idx, gt_idx, metric_value]]
        assigned_gt_dict = {'boxes': [], 'labels': []}

        if M == 0:
            neg_indices = list(range(N))
            return [], neg_indices, target_scores, assigned_gt_dict, []

        # 超参数 (YOLOv8 默认 alpha=1, beta=6)
        alpha = 1.0
        beta = 6.0
        
        # 1. 预处理：获取分类分数 (sigmoid)
        # 注意：cls_score_flat 是 logits，这里先转成概率
        # 虽然训练初期这是噪声，但随着训练进行，它会引导分配
        cls_pred = torch.sigmoid(cls_score_flat.squeeze(-1)) # [N]

        # 2. 计算该层所有点相对于所有 GT 的 "假设 IoU"
        # 我们假设该点预测的框完美贴合 GT 中心，或者直接计算点与 GT 的关系
        # 这里为了计算效率，我们计算：如果这个点负责预测，它能产生的最大可能 IoU
        # 即：以该点为中心，向四周延伸刚好覆盖 GT 时的 IoU
        # 简化版：直接计算 GT 框与该点所在感受野的 IoU，或者用高斯距离
        
        # 这里我们实现一个高效的近似：计算点到 GT 中心的距离，以及点是否在 GT 内
        # 并以此构造一个 cost 或者 iou 矩阵
        
        # 扩展维度以便广播计算: [N, 3] -> [N, 1, 3], [M, 6] -> [1, M, 6]
        centers_expanded = center_points.unsqueeze(1) # [N, 1, 3]
        gt_boxes_expanded = gt_boxes.unsqueeze(0)     # [1, M, 6]
        
        gz = gt_boxes_expanded[..., 0]
        gy = gt_boxes_expanded[..., 1]
        gx = gt_boxes_expanded[..., 2]
        gd = gt_boxes_expanded[..., 3]
        gh = gt_boxes_expanded[..., 4]
        gw = gt_boxes_expanded[..., 5]
        
        # 计算 GT 的 Corner 格式
        gt_zmin = gz - gd / 2
        gt_zmax = gz + gd / 2
        gt_ymin = gy - gh / 2
        gt_ymax = gy + gh / 2
        gt_xmin = gx - gw / 2
        gt_xmax = gx + gw / 2
        
        # 计算每个特征点到每个 GT 中心的 L1 距离
        # 注意：YOLOv8 实际上是用预测框和 GT 算 IoU，但在分配时我们还没有预测框
        # 所以这里我们用一个简单的代理：点在 GT 内的程度 + 距离的倒数
        # 或者更简单的：我们假设该点预测的框就是 GT 大小，以此算 IoU
        
        # 为了简化，我们先计算 Mask：点是否在 GT 内部或附近？
        # 这一步是为了过滤掉完全不可能的点，减少计算量并防止离谱的分配
        # 我们把 GT 稍微扩大一点 (1.5x) 来选候选点
        expand_ratio = 1.5
        mask_in_gt_neighborhood = (
            (centers_expanded[..., 0] >= gt_zmin - gd * (expand_ratio-1)/2) &
            (centers_expanded[..., 0] <= gt_zmax + gd * (expand_ratio-1)/2) &
            (centers_expanded[..., 1] >= gt_ymin - gh * (expand_ratio-1)/2) &
            (centers_expanded[..., 1] <= gt_ymax + gh * (expand_ratio-1)/2) &
            (centers_expanded[..., 2] >= gt_xmin - gw * (expand_ratio-1)/2) &
            (centers_expanded[..., 2] <= gt_xmax + gw * (expand_ratio-1)/2)
        ) # [N, M]

        # 调试：显示每层有多少点落在 GT 邻域内
        mask_any_gt = mask_in_gt_neighborhood.any(dim=1)  # [N] 每个点是否在某个GT邻域内
        in_neighborhood_count = mask_any_gt.sum().item()
        
        # 打印GT的大小范围供参考（debug模式）
        if debug:
            gt_sizes = torch.stack([gd.mean(), gh.mean(), gw.mean()]).cpu().numpy()
            feat_stride = int(96 / (N ** (1/3)))
            print(f"    [DEBUG TAA Lvl{level_idx}] N={N}, stride={feat_stride}, "
                  f"GT_size: z={gt_sizes[0]:.1f}, y={gt_sizes[1]:.1f}, x={gt_sizes[2]:.1f}")
            print(f"    [DEBUG TAA Lvl{level_idx}] points_in_GT_neighborhood={in_neighborhood_count}/{N} ({in_neighborhood_count/N*100:.1f}%)")

        # 3. 计算对齐度矩阵 Alignment Metric Matrix
        # 初始化全 0
        alignment_metric = torch.zeros((N, M), device=center_points.device)
        
        # 只对 GT 附近的点计算 Metric
        if mask_in_gt_neighborhood.any():
            # 计算距离惩罚项 (越近越好) -> 转换为类似 IoU 的 0~1 分数
            dist_z = (centers_expanded[..., 0] - gz).abs()
            dist_y = (centers_expanded[..., 1] - gy).abs()
            dist_x = (centers_expanded[..., 2] - gx).abs()
            
            # 归一化距离：0表示重合，1表示在 GT 边界，>1表示在外面
            norm_dist_z = dist_z / (gd / 2 + 1e-6)
            norm_dist_y = dist_y / (gh / 2 + 1e-6)
            norm_dist_x = dist_x / (gw / 2 + 1e-6)
            max_norm_dist = torch.max(torch.stack([norm_dist_z, norm_dist_y, norm_dist_x], dim=-1), dim=-1)[0]
            
            # 构造一个伪 IoU (Position Score): 1 最好，0 最差
            # 这比直接算 IoU 快很多，且适合 Anchor-Free
            position_score = 1.0 / (max_norm_dist + 1e-6)
            position_score = torch.clamp(position_score, min=0.0, max=1.0)
            
            # 结合分类分数
            # cls_pred: [N] -> [N, 1] -> [N, M]
            cls_score_expanded = cls_pred.unsqueeze(1).expand(-1, M)
            
            # Metric = s^alpha * iou^beta
            alignment_metric = (torch.pow(cls_score_expanded, alpha) * 
                                torch.pow(position_score, beta))
            
            # 只保留邻域内的 Metric，其余设为 -inf 或 0
            alignment_metric[~mask_in_gt_neighborhood] = -float('inf')

        # 4. 为每个 GT 选择 Top-K 个点
        # 注意：这里我们是在当前层内选 Top-K，而不是跨所有层
        for gt_idx in range(M):
            gt_box = gt_boxes[gt_idx]
            gt_label = gt_labels[gt_idx]
            
            # 获取当前 GT 对应的 Metric 列
            metric_per_gt = alignment_metric[:, gt_idx]
            
            # 排序，取前 max_pos_per_gt 个
            topk_metrics, topk_indices = torch.topk(metric_per_gt, k=min(max_pos_per_gt, N), largest=True)
            
            # 过滤掉无效的分配 (Metric <= 0 的不要)
            valid_mask = topk_metrics > 1e-6
            selected_indices = topk_indices[valid_mask].tolist()
            
            if len(selected_indices) > 0:
                for idx in selected_indices:
                    # 记录：(位置索引, GT索引, 对齐度分数)
                    pos_to_gt_mapping.append((idx, gt_idx, alignment_metric[idx, gt_idx].item()))
                
                assigned_gt_dict['boxes'].append(gt_box)
                assigned_gt_dict['labels'].append(gt_label)
                
                if debug:
                    print(f"    [DEBUG TAA Lvl{level_idx}] GT{gt_idx} -> "
                          f"selected {len(selected_indices)} points (topk metric: {topk_metrics[0].item():.3f})")

        # 5. 去重 (One-to-Many -> Many-to-One)
        # 如果一个特征点被分配给了多个 GT，只保留 Metric 最高的那个配对
        unique_assignments = {} # {pos_idx: (gt_idx, max_metric)}
        for idx, gt_idx, metric in pos_to_gt_mapping:
            if idx not in unique_assignments or metric > unique_assignments[idx][1]:
                unique_assignments[idx] = (gt_idx, metric)

        # 6. 构建最终输出
        final_pos_indices = []
        final_gt_indices = []
        for idx, (gt_idx, _) in unique_assignments.items():
            final_pos_indices.append(idx)
            final_gt_indices.append(gt_idx)
            target_scores[idx] = 1.0 # 目标分数设为 1

        pos_set = set(final_pos_indices)
        all_neg_before_sample = [i for i in range(N) if i not in pos_set]
        
        # 负样本采样
        neg_indices = self._sample_negatives(
            all_neg_before_sample, cls_score_flat, target_scores
        )
        
        # 调试：显示采样前后对比（debug模式）
        if debug and not hasattr(self, '_debug_assign_printed'):
            print(f"  [DEBUG assign] Total N={N}, pos={len(final_pos_indices)}, "
                  f"neg_before_sample={len(all_neg_before_sample)}, neg_after_sample={len(neg_indices)}")
            self._debug_assign_printed = True

        # 准备 GT boxes list
        gt_boxes_for_loss = [gt_boxes[gt_idx] for gt_idx in final_gt_indices]

        return final_pos_indices, neg_indices, target_scores, assigned_gt_dict, gt_boxes_for_loss

    @staticmethod
    def _points_inside_boxes(center_points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Return which feature points fall inside any center-size box."""
        if boxes is None or boxes.numel() == 0:
            return torch.zeros(center_points.shape[0], dtype=torch.bool, device=center_points.device)

        boxes = boxes.to(device=center_points.device, dtype=center_points.dtype)
        points = center_points.unsqueeze(1)
        centers = boxes[:, :3].unsqueeze(0)
        half_sizes = boxes[:, 3:].unsqueeze(0) / 2
        return ((points >= centers - half_sizes) & (points <= centers + half_sizes)).all(dim=2).any(dim=1)

    def _task_aligned_assign_global(
        self,
        center_points_list: List[torch.Tensor],
        cls_score_flat_list: List[torch.Tensor],
        reg_pred_flat_list: List[torch.Tensor],
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        ignored_boxes: torch.Tensor,
        image_shape: Tuple[int, int, int],
        debug: bool = False,
        image_path: str = 'unknown',
        max_pos_per_gt: int = 3,
    ) -> List[Dict[str, object]]:
        level_results = []
        level_offsets = []
        total_points = 0
        for points in center_points_list:
            level_offsets.append(total_points)
            total_points += points.shape[0]
            level_results.append({
                'pos_indices': [],
                'neg_indices': [],
                'ignore_indices': [],
                'target_scores': torch.zeros(points.shape[0], device=points.device),
                'assigned_gt': {'boxes': [], 'labels': []},
                'gt_boxes_for_loss': [],
            })

        if gt_boxes.numel() == 0:
            for level_idx, points in enumerate(center_points_list):
                ignored_mask = self._points_inside_boxes(points, ignored_boxes)
                ignore_set = set(torch.where(ignored_mask)[0].tolist())
                level_results[level_idx]['ignore_indices'] = sorted(ignore_set)
                all_neg_before_sample = [
                    i for i in range(points.shape[0])
                    if i not in ignore_set
                ]
                level_results[level_idx]['neg_indices'] = self._sample_negatives(
                    all_neg_before_sample,
                    cls_score_flat_list[level_idx],
                    level_results[level_idx]['target_scores'],
                )
            return level_results

        center_points = torch.cat(center_points_list, dim=0)
        cls_score_flat = torch.cat(cls_score_flat_list, dim=0)
        reg_pred_flat = torch.cat(reg_pred_flat_list, dim=0)

        N = center_points.shape[0]
        M = gt_boxes.shape[0]
        alpha = self.assign_alpha
        beta = self.assign_beta
        cls_pred = torch.sigmoid(cls_score_flat.squeeze(-1))
        assignment_quality = self.assignment_quality
        if assignment_quality == "pred_iou" and not self.assign_use_pred_iou:
            assignment_quality = "distance"
        assign_one_level = self.assign_one_level
        if self.assignment_mode == "multi_level":
            assign_one_level = False
        elif self.assignment_mode == "one_level":
            assign_one_level = True

        centers_expanded = center_points.unsqueeze(1)
        gt_boxes_expanded = gt_boxes.unsqueeze(0)

        gz = gt_boxes_expanded[..., 0]
        gy = gt_boxes_expanded[..., 1]
        gx = gt_boxes_expanded[..., 2]
        gd = gt_boxes_expanded[..., 3]
        gh = gt_boxes_expanded[..., 4]
        gw = gt_boxes_expanded[..., 5]

        gt_zmin = gz - gd / 2
        gt_zmax = gz + gd / 2
        gt_ymin = gy - gh / 2
        gt_ymax = gy + gh / 2
        gt_xmin = gx - gw / 2
        gt_xmax = gx + gw / 2

        mask_in_gt = (
            (centers_expanded[..., 0] >= gt_zmin) &
            (centers_expanded[..., 0] <= gt_zmax) &
            (centers_expanded[..., 1] >= gt_ymin) &
            (centers_expanded[..., 1] <= gt_ymax) &
            (centers_expanded[..., 2] >= gt_xmin) &
            (centers_expanded[..., 2] <= gt_xmax)
        )

        # Pred-IoU assignment follows YOLO-style center-in-GT candidate filtering.
        # The expanded neighborhood is retained for the other assignment ablations.
        if assignment_quality == "pred_iou":
            mask_in_gt_neighborhood = mask_in_gt
        else:
            expand_ratio = self.assign_expand_ratio
            mask_in_gt_neighborhood = (
                (centers_expanded[..., 0] >= gt_zmin - gd * (expand_ratio - 1) / 2) &
                (centers_expanded[..., 0] <= gt_zmax + gd * (expand_ratio - 1) / 2) &
                (centers_expanded[..., 1] >= gt_ymin - gh * (expand_ratio - 1) / 2) &
                (centers_expanded[..., 1] <= gt_ymax + gh * (expand_ratio - 1) / 2) &
                (centers_expanded[..., 2] >= gt_xmin - gw * (expand_ratio - 1) / 2) &
                (centers_expanded[..., 2] <= gt_xmax + gw * (expand_ratio - 1) / 2)
            )

        if debug:
            in_neighborhood_count = mask_in_gt_neighborhood.any(dim=1).sum().item()
            gt_sizes = torch.stack([gd.mean(), gh.mean(), gw.mean()]).cpu().numpy()
            print(f"    [DEBUG TAA Global] N={N}, GT_size: z={gt_sizes[0]:.1f}, y={gt_sizes[1]:.1f}, x={gt_sizes[2]:.1f}")
            print(f"    [DEBUG TAA Global] points_in_GT_neighborhood={in_neighborhood_count}/{N} ({in_neighborhood_count/N*100:.1f}%)")

        distance_metric = torch.zeros((N, M), device=center_points.device)
        alignment_metric = torch.zeros((N, M), device=center_points.device)
        pred_iou = None
        if mask_in_gt_neighborhood.any():
            dist_z = (centers_expanded[..., 0] - gz).abs()
            dist_y = (centers_expanded[..., 1] - gy).abs()
            dist_x = (centers_expanded[..., 2] - gx).abs()

            norm_dist_z = dist_z / (gd / 2 + 1e-6)
            norm_dist_y = dist_y / (gh / 2 + 1e-6)
            norm_dist_x = dist_x / (gw / 2 + 1e-6)
            max_norm_dist = torch.max(torch.stack([norm_dist_z, norm_dist_y, norm_dist_x], dim=-1), dim=-1)[0]

            position_score = 1.0 / (max_norm_dist + 1e-6)
            position_score = torch.clamp(position_score, min=0.0, max=1.0)
            cls_score_expanded = cls_pred.unsqueeze(1).expand(-1, M)
            distance_metric = torch.pow(cls_score_expanded, alpha) * torch.pow(position_score, beta)
            distance_metric[~mask_in_gt_neighborhood] = -float('inf')

            if assignment_quality == "center_topk":
                alignment_metric = -max_norm_dist
                alignment_metric[~mask_in_gt_neighborhood] = -float('inf')
            elif assignment_quality == "pred_iou":
                if self.use_dfl:
                    decoded_reg = self.dfl_loss.decode_dist(reg_pred_flat)
                else:
                    decoded_reg = reg_pred_flat
                pred_boxes = self._decode_boxes_yolov8(center_points, decoded_reg, image_shape)
                gt_boxes_corner = self._center_size_to_corners(gt_boxes)
                pred_iou = self._compute_pairwise_iou_3d(pred_boxes, gt_boxes_corner)
                pred_iou = pred_iou.clamp(min=0.0, max=1.0)
                alignment_metric = torch.pow(cls_score_expanded, alpha) * torch.pow(pred_iou, beta)
                alignment_metric[~mask_in_gt_neighborhood] = -float('inf')
            elif assignment_quality == "distance":
                alignment_metric = distance_metric
            else:
                raise ValueError(
                    f"Unsupported assignment_quality={assignment_quality!r}. "
                    "Expected 'distance', 'pred_iou', or 'center_topk'."
                )

        unique_assignments = {}
        for gt_idx in range(M):
            metric_per_gt = alignment_metric[:, gt_idx]
            if assign_one_level:
                if assignment_quality == "center_topk":
                    valid_metric_mask = torch.isfinite(metric_per_gt)
                else:
                    valid_metric_mask = metric_per_gt > 1e-6
                if valid_metric_mask.any():
                    best_global_idx = torch.argmax(
                        torch.where(valid_metric_mask, metric_per_gt, torch.full_like(metric_per_gt, -float('inf')))
                    ).item()
                    selected_level_idx = max(
                        idx for idx, offset in enumerate(level_offsets) if offset <= best_global_idx
                    )
                    level_start = level_offsets[selected_level_idx]
                    level_end = (
                        level_offsets[selected_level_idx + 1]
                        if selected_level_idx + 1 < len(level_offsets)
                        else total_points
                    )
                    level_mask = torch.zeros_like(metric_per_gt, dtype=torch.bool)
                    level_mask[level_start:level_end] = True
                    metric_per_gt = torch.where(
                        level_mask,
                        metric_per_gt,
                        torch.full_like(metric_per_gt, -float('inf')),
                    )
            topk_metrics, topk_indices = torch.topk(metric_per_gt, k=min(max_pos_per_gt, N), largest=True)
            if assignment_quality == "center_topk":
                valid_mask = torch.isfinite(topk_metrics)
            else:
                valid_mask = topk_metrics > 1e-6
            selected_indices = topk_indices[valid_mask].tolist()
            selected_from_pred_iou = assignment_quality == "pred_iou"
            if not selected_indices and assignment_quality == "pred_iou":
                fallback_metric_per_gt = distance_metric[:, gt_idx]
                if assign_one_level:
                    valid_metric_mask = fallback_metric_per_gt > 1e-6
                    if valid_metric_mask.any():
                        best_global_idx = torch.argmax(
                            torch.where(
                                valid_metric_mask,
                                fallback_metric_per_gt,
                                torch.full_like(fallback_metric_per_gt, -float('inf')),
                            )
                        ).item()
                        selected_level_idx = max(
                            idx for idx, offset in enumerate(level_offsets) if offset <= best_global_idx
                        )
                        level_start = level_offsets[selected_level_idx]
                        level_end = (
                            level_offsets[selected_level_idx + 1]
                            if selected_level_idx + 1 < len(level_offsets)
                            else total_points
                        )
                        level_mask = torch.zeros_like(fallback_metric_per_gt, dtype=torch.bool)
                        level_mask[level_start:level_end] = True
                        fallback_metric_per_gt = torch.where(
                            level_mask,
                            fallback_metric_per_gt,
                            torch.full_like(fallback_metric_per_gt, -float('inf')),
                        )
                topk_metrics, topk_indices = torch.topk(
                    fallback_metric_per_gt,
                    k=min(max_pos_per_gt, N),
                    largest=True,
                )
                valid_mask = topk_metrics > 1e-6
                selected_indices = topk_indices[valid_mask].tolist()
                metric_per_gt = fallback_metric_per_gt
                selected_from_pred_iou = False
            if debug:
                print(f"    [DEBUG TAA Global] GT{gt_idx} -> selected {len(selected_indices)} points")
            selected_target_scores = {}
            if (
                self.assign_soft_target_scores
                and selected_from_pred_iou
                and pred_iou is not None
                and selected_indices
            ):
                selected_tensor = torch.as_tensor(selected_indices, dtype=torch.long, device=center_points.device)
                selected_metrics = metric_per_gt[selected_tensor]
                max_metric = selected_metrics.max().clamp(min=1e-12)
                max_iou = pred_iou[selected_tensor, gt_idx].max().clamp(min=0.0, max=1.0)
                normalized_scores = (selected_metrics / max_metric * max_iou).clamp(min=0.0, max=1.0)
                selected_target_scores = {
                    idx: float(score.item())
                    for idx, score in zip(selected_indices, normalized_scores)
                }
            for idx in selected_indices:
                metric = metric_per_gt[idx].item()
                target_score = selected_target_scores.get(idx, 1.0)
                if idx not in unique_assignments or metric > unique_assignments[idx][1]:
                    unique_assignments[idx] = (gt_idx, metric, target_score)

        for global_idx, (gt_idx, _, target_score) in unique_assignments.items():
            level_idx = 0
            local_idx = global_idx
            for idx, offset in enumerate(level_offsets):
                next_offset = level_offsets[idx + 1] if idx + 1 < len(level_offsets) else total_points
                if offset <= global_idx < next_offset:
                    level_idx = idx
                    local_idx = global_idx - offset
                    break
            level_results[level_idx]['pos_indices'].append(local_idx)
            level_results[level_idx]['target_scores'][local_idx] = target_score
            level_results[level_idx]['assigned_gt']['boxes'].append(gt_boxes[gt_idx])
            level_results[level_idx]['assigned_gt']['labels'].append(gt_labels[gt_idx])
            level_results[level_idx]['gt_boxes_for_loss'].append(gt_boxes[gt_idx])

        for level_idx, result in enumerate(level_results):
            pos_set = set(result['pos_indices'])
            level_points = center_points_list[level_idx]
            level_centers_expanded = level_points.unsqueeze(1)
            ignore_expand_ratio = self.assign_ignore_expand_ratio if assignment_quality == "pred_iou" else 1.0
            ignore_gt_mask = (
                (level_centers_expanded[..., 0] >= gt_zmin.squeeze(0) - gd.squeeze(0) * (ignore_expand_ratio - 1) / 2) &
                (level_centers_expanded[..., 0] <= gt_zmax.squeeze(0) + gd.squeeze(0) * (ignore_expand_ratio - 1) / 2) &
                (level_centers_expanded[..., 1] >= gt_ymin.squeeze(0) - gh.squeeze(0) * (ignore_expand_ratio - 1) / 2) &
                (level_centers_expanded[..., 1] <= gt_ymax.squeeze(0) + gh.squeeze(0) * (ignore_expand_ratio - 1) / 2) &
                (level_centers_expanded[..., 2] >= gt_xmin.squeeze(0) - gw.squeeze(0) * (ignore_expand_ratio - 1) / 2) &
                (level_centers_expanded[..., 2] <= gt_xmax.squeeze(0) + gw.squeeze(0) * (ignore_expand_ratio - 1) / 2)
            )
            ignored_box_mask = self._points_inside_boxes(level_points, ignored_boxes)
            high_iou_mask = torch.zeros(level_points.shape[0], dtype=torch.bool, device=level_points.device)
            if self.neg_iou_thr >= 0:
                if pred_iou is None:
                    if self.use_dfl:
                        decoded_reg = self.dfl_loss.decode_dist(reg_pred_flat)
                    else:
                        decoded_reg = reg_pred_flat
                    pred_boxes = self._decode_boxes_yolov8(center_points, decoded_reg, image_shape)
                    gt_boxes_corner = self._center_size_to_corners(gt_boxes)
                    pred_iou = self._compute_pairwise_iou_3d(pred_boxes, gt_boxes_corner)
                level_start = level_offsets[level_idx]
                level_end = level_start + level_points.shape[0]
                high_iou_mask = pred_iou[level_start:level_end].max(dim=1).values >= self.neg_iou_thr
            ignore_set = (
                set(torch.where(ignore_gt_mask.any(dim=1))[0].tolist()) |
                set(torch.where(ignored_box_mask)[0].tolist()) |
                set(torch.where(high_iou_mask)[0].tolist())
            ) - pos_set
            result['ignore_indices'] = sorted(ignore_set)
            all_neg_before_sample = [
                i for i in range(result['target_scores'].shape[0])
                if i not in pos_set and i not in ignore_set
            ]
            result['neg_indices'] = self._sample_negatives(
                all_neg_before_sample,
                cls_score_flat_list[level_idx],
                result['target_scores'],
            )
            if debug and not hasattr(self, '_debug_assign_printed'):
                print(
                    f"  [DEBUG assign global->level{level_idx}] N={result['target_scores'].shape[0]}, "
                    f"pos={len(result['pos_indices'])}, neg_before_sample={len(all_neg_before_sample)}, "
                    f"ignore={len(result['ignore_indices'])}, neg_after_sample={len(result['neg_indices'])}"
                )

        if debug and not hasattr(self, '_debug_assign_printed'):
            self._debug_assign_printed = True

        return level_results

    def _sample_negatives(
        self,
        neg_indices: List[int],
        cls_score_flat: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> List[int]:
        """
        OHEM + 随机采样负样本

        采样逻辑：
        1. 从所有负样本中，按预测得分从高到低排序
        2. 取前 N 个作为困难负样本（分数高 = 最像目标 = 最难）
        3. 从剩下的负样本中随机采样
        4. 合并 → 得到最终负样本集合

        Args:
            neg_indices: 所有负样本索引
            cls_score_flat: 分类预测分数
            target_scores: 目标分数

        Returns:
            采样后的负样本索引
        """
        if len(neg_indices) == 0:
            return neg_indices

        N_neg = len(neg_indices)

        # 计算困难样本数量和随机采样数量
        hard_k = max(1, int(N_neg * self.neg_sample_hard_ratio))
        random_k = max(1, int(N_neg * self.neg_sample_random_ratio))

        # 获取所有负样本的预测分数
        neg_scores = cls_score_flat[neg_indices].squeeze(-1)

        # 按预测得分从高到低排序（分数高 = 最像正样本 = 最难）
        sorted_indices = torch.argsort(neg_scores, descending=True)

        # 步骤2：从高到低取前 hard_k 个作为困难负样本
        if hard_k > 0 and hard_k < len(neg_indices):
            hard_idx_in_sorted = sorted_indices[:hard_k].tolist()
            hard_indices = [neg_indices[i] for i in hard_idx_in_sorted]
        else:
            hard_indices = neg_indices

        # 步骤3：从剩下的负样本中随机采样
        used_indices = set(hard_indices)
        remaining_indices = [idx for idx in neg_indices if idx not in used_indices]

        if random_k > 0 and len(remaining_indices) > 0:
            random_k = min(random_k, len(remaining_indices))
            random_indices = random.sample(remaining_indices, random_k)
        else:
            random_indices = remaining_indices[:random_k] if remaining_indices else []

        # 步骤4：合并
        final_neg_indices = hard_indices + random_indices
        
        return final_neg_indices

    def _decode_boxes_yolov8(
        self,
        center_points: torch.Tensor,
        reg_pred: torch.Tensor,
        image_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        YOLOv8风格框解码

        使用exp()确保回归输出为正值，然后加上中心点坐标得到corner格式

        Args:
            center_points: [N, 3] 特征点中心坐标
            reg_pred: [N, 6] 预测的6个边界距离 (z1, y1, x1, z2, y2, x2)，原始输出
            image_shape: 原图尺寸

        Returns:
            boxes: [N, 6] corner格式 (z_min, y_min, x_min, z_max, y_max, x_max)
        """
        # 直接使用输出作为距离（不需要exp，因为GT就是体素距离）
        z1 = reg_pred[:, 0].abs()
        z2 = reg_pred[:, 3].abs()
        y1 = reg_pred[:, 1].abs()
        y2 = reg_pred[:, 4].abs()
        x1 = reg_pred[:, 2].abs()
        x2 = reg_pred[:, 5].abs()

        z_min = (center_points[:, 0] - z1).clamp(min=0, max=image_shape[0])
        z_max = (center_points[:, 0] + z2).clamp(min=0, max=image_shape[0])
        y_min = (center_points[:, 1] - y1).clamp(min=0, max=image_shape[1])
        y_max = (center_points[:, 1] + y2).clamp(min=0, max=image_shape[1])
        x_min = (center_points[:, 2] - x1).clamp(min=0, max=image_shape[2])
        x_max = (center_points[:, 2] + x2).clamp(min=0, max=image_shape[2])

        return torch.stack([z_min, y_min, x_min, z_max, y_max, x_max], dim=1)

    def _center_size_to_corners(self, boxes: torch.Tensor) -> torch.Tensor:
        """将中心+尺寸格式转换为corner格式"""
        if boxes.numel() == 0:
            return boxes

        z = boxes[:, 0]
        y = boxes[:, 1]
        x = boxes[:, 2]
        d = boxes[:, 3]
        h = boxes[:, 4]
        w = boxes[:, 5]

        z_min = z - d / 2
        z_max = z + d / 2
        y_min = y - h / 2
        y_max = y + h / 2
        x_min = x - w / 2
        x_max = x + w / 2

        return torch.stack([z_min, y_min, x_min, z_max, y_max, x_max], dim=1)

    def _compute_dfl_ciou_loss(
        self,
        center_points: torch.Tensor,
        pred_dist: torch.Tensor,
        target_boxes: torch.Tensor,
        image_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算DFL + CIoU组合损失

        Args:
            center_points: [N, 3] 特征点中心坐标
            pred_dist: [N, 6, num_bins] 预测的分布概率
            target_boxes: [N, 6] 目标框 (corner格式)
            image_shape: 原图尺寸

        Returns:
            dfl_loss: DFL损失 (scalar)
            ciou_loss: CIoU损失 (scalar)
        """
        N, C, B = pred_dist.shape

        # 1. 从分布解码预测距离值
        target_dists = self.dfl_loss.decode_dist(pred_dist)  # [N, 6]

        # 2. 解码预测框
        pred_boxes = self._decode_boxes_from_dist(center_points, target_dists, image_shape)

        # 3. 计算GT的距离值（从corner格式）
        gt_dists = torch.zeros_like(target_boxes)
        gt_dists[:, 0] = center_points[:, 0] - target_boxes[:, 0]  # z1
        gt_dists[:, 1] = center_points[:, 1] - target_boxes[:, 1]  # y1
        gt_dists[:, 2] = center_points[:, 2] - target_boxes[:, 2]  # x1
        gt_dists[:, 3] = target_boxes[:, 3] - center_points[:, 0]  # z2
        gt_dists[:, 4] = target_boxes[:, 4] - center_points[:, 1]  # y2
        gt_dists[:, 5] = target_boxes[:, 5] - center_points[:, 2]  # x2
        gt_dists = gt_dists.abs()

        # 4. DFL损失
        dfl_loss = self.dfl_loss(pred_dist, gt_dists)

        # 5. CIoU损失（使用解码后的预测框和GT框）
        ciou_loss = self._compute_bbox_loss(pred_boxes, target_boxes)

        return dfl_loss, ciou_loss

    def _compute_bbox_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "diou":
            return self._compute_diou_loss(pred_boxes, target_boxes)
        return self._compute_ciou_loss(pred_boxes, target_boxes)

    def _compute_diou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        if pred_boxes.numel() == 0:
            return torch.tensor([], device=pred_boxes.device)

        iou = self._compute_iou_3d(pred_boxes, target_boxes)
        pred_center = (pred_boxes[:, :3] + pred_boxes[:, 3:]) / 2
        target_center = (target_boxes[:, :3] + target_boxes[:, 3:]) / 2
        center_dist_sq = ((pred_center - target_center) ** 2).sum(dim=1)
        enclose_min = torch.min(pred_boxes[:, :3], target_boxes[:, :3])
        enclose_max = torch.max(pred_boxes[:, 3:], target_boxes[:, 3:])
        enclose_dist_sq = ((enclose_max - enclose_min) ** 2).sum(dim=1)
        diou = iou - center_dist_sq / (enclose_dist_sq + 1e-6)
        return 1.0 - torch.clamp(diou, min=-1.0, max=1.0)

    def _decode_boxes_from_dist(
        self,
        center_points: torch.Tensor,
        pred_dist: torch.Tensor,
        image_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        从DFL解码的距离值解码为角点格式框

        Args:
            center_points: [N, 3] 特征点中心坐标
            pred_dist: [N, 6] 解码后的距离值
            image_shape: 原图尺寸

        Returns:
            boxes: [N, 6] corner格式
        """
        z1 = pred_dist[:, 0].abs()
        z2 = pred_dist[:, 3].abs()
        y1 = pred_dist[:, 1].abs()
        y2 = pred_dist[:, 4].abs()
        x1 = pred_dist[:, 2].abs()
        x2 = pred_dist[:, 5].abs()

        z_min = (center_points[:, 0] - z1).clamp(min=0, max=image_shape[0])
        z_max = (center_points[:, 0] + z2).clamp(min=0, max=image_shape[0])
        y_min = (center_points[:, 1] - y1).clamp(min=0, max=image_shape[1])
        y_max = (center_points[:, 1] + y2).clamp(min=0, max=image_shape[1])
        x_min = (center_points[:, 2] - x1).clamp(min=0, max=image_shape[2])
        x_max = (center_points[:, 2] + x2).clamp(min=0, max=image_shape[2])

        return torch.stack([z_min, y_min, x_min, z_max, y_max, x_max], dim=1)

    def _compute_ciou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        计算3D CIoU损失 (Complete IoU)

        CIoU = IoU - 中心距离惩罚 - 长宽比惩罚

        Args:
            pred_boxes: [N, 6] 预测框 (corner格式: z_min, y_min, x_min, z_max, y_max, x_max)
            target_boxes: [N, 6] 目标框 (corner格式)

        Returns:
            loss: [N] 每个样本的CIoU损失，范围 [0, 2]
        """
        if pred_boxes.numel() == 0:
            return torch.tensor([], device=pred_boxes.device)

        # ===== IoU计算 =====
        iou = self._compute_iou_3d(pred_boxes, target_boxes)

        # ===== 中心点距离 (DIoU部分) =====
        pred_center = (pred_boxes[:, :3] + pred_boxes[:, 3:]) / 2
        target_center = (target_boxes[:, :3] + target_boxes[:, 3:]) / 2
        center_dist_sq = ((pred_center - target_center) ** 2).sum(dim=1)

        # 最小包围框对角线
        enclose_min = torch.min(pred_boxes[:, :3], target_boxes[:, :3])
        enclose_max = torch.max(pred_boxes[:, 3:], target_boxes[:, 3:])
        enclose_dist_sq = ((enclose_max - enclose_min) ** 2).sum(dim=1)

        # ===== 长宽比惩罚 (CIoU新增) =====
        # 提取尺寸 (corner -> size)
        pred_d = (pred_boxes[:, 3] - pred_boxes[:, 0]).clamp_min(1e-6)  # depth
        pred_h = (pred_boxes[:, 4] - pred_boxes[:, 1]).clamp_min(1e-6)  # height
        pred_w = (pred_boxes[:, 5] - pred_boxes[:, 2]).clamp_min(1e-6)  # width

        target_d = (target_boxes[:, 3] - target_boxes[:, 0]).clamp_min(1e-6)
        target_h = (target_boxes[:, 4] - target_boxes[:, 1]).clamp_min(1e-6)
        target_w = (target_boxes[:, 5] - target_boxes[:, 2]).clamp_min(1e-6)

        # 避免除零
        # 3D 长宽比: 使用 d/h 和 d/w (类似于2D的 w/h)
        # v = (4/π²) × [(arctan(d/h) - arctan(d'/h'))² + (arctan(d/w) - arctan(d'/w'))²]
        v = (4.0 / (math.pi ** 2)) * (
            (torch.atan(pred_d / pred_h) - torch.atan(target_d / target_h)) ** 2 +
            (torch.atan(pred_d / pred_w) - torch.atan(target_d / target_w)) ** 2
        )

        # 平衡参数 α
        alpha = v / (1.0 - iou + v + 1e-6)

        # ===== CIoU =====
        # CIoU = IoU - (d²/c²) - α×v
        ciou = iou - center_dist_sq / (enclose_dist_sq + 1e-6) - alpha * v

        # 限制范围 [-1, 1]
        ciou = torch.clamp(ciou, min=-1.0, max=1.0)

        return 1.0 - ciou

    def _compute_iou_3d(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """计算3D IoU"""
        inter_min = torch.max(boxes1[:, :3], boxes2[:, :3])
        inter_max = torch.min(boxes1[:, 3:], boxes2[:, 3:])
        inter_size = (inter_max - inter_min).clamp(min=0)
        inter_vol = inter_size[:, 0] * inter_size[:, 1] * inter_size[:, 2]

        vol1 = (boxes1[:, 3] - boxes1[:, 0]) * (boxes1[:, 4] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 2])
        vol2 = (boxes2[:, 3] - boxes2[:, 0]) * (boxes2[:, 4] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 2])

        union_vol = vol1 + vol2 - inter_vol

        return inter_vol / (union_vol + 1e-6)

    def _compute_pairwise_iou_3d(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise 3D IoU between boxes1 [N, 6] and boxes2 [M, 6]."""
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

        boxes1_expanded = boxes1.unsqueeze(1)
        boxes2_expanded = boxes2.unsqueeze(0)
        inter_min = torch.max(boxes1_expanded[..., :3], boxes2_expanded[..., :3])
        inter_max = torch.min(boxes1_expanded[..., 3:], boxes2_expanded[..., 3:])
        inter_size = (inter_max - inter_min).clamp(min=0)
        inter_vol = inter_size[..., 0] * inter_size[..., 1] * inter_size[..., 2]

        size1 = (boxes1[:, 3:] - boxes1[:, :3]).clamp(min=0)
        size2 = (boxes2[:, 3:] - boxes2[:, :3]).clamp(min=0)
        vol1 = (size1[:, 0] * size1[:, 1] * size1[:, 2]).unsqueeze(1)
        vol2 = (size2[:, 0] * size2[:, 1] * size2[:, 2]).unsqueeze(0)
        union_vol = vol1 + vol2 - inter_vol

        return inter_vol / (union_vol + 1e-6)


class AnchorFreePostProcess3D(nn.Module):
    """
    YOLOv8风格无锚框3D检测后处理

    流程：
    1. 分类分数阈值过滤（分类分数直接作为检测置信度）
    2. 3D NMS去重

    关键修改（方案A）：
    - 删除独立的置信度分支
    - 分类分数直接作为检测分数
    """

    def __init__(
        self,
        score_threshold: float = 0.25,
        nms_iou_threshold: float = 0.5,
        max_detections: int = 100,
        num_classes: int = 1,
        dfl_range_low: float = 0.0,
        dfl_range_high: float = 48.0,
    ):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections
        self.num_classes = num_classes
        self.dfl_range_low = dfl_range_low
        self.dfl_range_high = dfl_range_high

        # 尝试导入DFL模块
        try:
            from .losses.dfl_loss import DFLLoss
            self.dfl_loss = DFLLoss(
                range_low=dfl_range_low,
                range_high=dfl_range_high,
                fine_bins=8,
                coarse_bins=10,
            )
            self.has_dfl = True
        except ImportError:
            self.dfl_loss = None
            self.has_dfl = False

    def forward(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        feature_sizes: List[List[int]],
        image_shape: Tuple[int, int, int],
    ) -> Dict[str, List[torch.Tensor]]:
        """
        后处理

        Args:
            predictions: 检测头输出 (包含cls_scores, reg_preds)
            feature_sizes: 特征图尺寸
            image_shape: 原图尺寸

        Returns:
            检测结果
        """
        cls_scores = predictions['cls_scores']
        reg_preds = predictions['reg_preds']

        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]

        # 使用DFLLoss模块中的num_bins值
        dfl_bins = self.dfl_loss.num_bins if self.has_dfl else 6

        all_boxes = []
        all_scores = []
        all_labels = []

        for b in range(batch_size):
            boxes_list = []
            scores_list = []
            labels_list = []

            for level_idx in range(len(cls_scores)):
                cls_score = cls_scores[level_idx][b]   # [C, D, H, W]
                reg_pred = reg_preds[level_idx][b]      # [6, D, H, W] 或 [6*18, D, H, W]
                feat_size = feature_sizes[level_idx]

                C = cls_score.shape[0]

                # 展平
                cls_score_flat = cls_score.permute(1, 2, 3, 0).reshape(-1, C)

                # 判断是否是DFL模式
                if self.has_dfl and reg_pred.shape[0] > 6:
                    # DFL模式: [6*18, D, H, W] -> [D, H, W, 6, 18] -> [N, 6, 18]
                    reg_pred_flat = reg_pred.permute(1, 2, 3, 0).reshape(feat_size[0]*feat_size[1]*feat_size[2], 6, dfl_bins)
                    # 从分布解码距离值
                    reg_pred_decoded = self.dfl_loss.decode_dist(reg_pred_flat)  # [N, 6]
                else:
                    # 普通模式
                    reg_pred_flat = reg_pred.permute(1, 2, 3, 0).reshape(-1, 6)
                    reg_pred_decoded = reg_pred_flat

                # 生成中心点网格
                center_points = self._generate_grid(feat_size, image_shape, device)

                # 解码预测框
                pred_boxes = self._decode_boxes(center_points, reg_pred_decoded, image_shape)

                # 分类分数直接作为检测分数（与YOLOv8一致）
                detection_scores = torch.sigmoid(cls_score_flat.squeeze(-1))

                # 分类分数过滤
                if C > 1:
                    # 多类别：选择最高类别的分数
                    cls_label_final = cls_score_flat.argmax(dim=-1)
                else:
                    cls_label_final = torch.zeros_like(detection_scores, dtype=torch.long)

                keep_mask = detection_scores > self.score_threshold
                if keep_mask.any():
                    boxes_list.append(pred_boxes[keep_mask])
                    scores_list.append(detection_scores[keep_mask])
                    labels_list.append(cls_label_final[keep_mask])

            # 合并所有尺度
            if boxes_list:
                all_level_boxes = torch.cat(boxes_list, dim=0)
                all_level_scores = torch.cat(scores_list, dim=0)
                all_level_labels = torch.cat(labels_list, dim=0)

                # 3D NMS
                keep_indices = self._nms_3d(
                    all_level_boxes, all_level_scores, all_level_labels
                )

                final_boxes = all_level_boxes[keep_indices]
                final_scores = all_level_scores[keep_indices]
                final_labels = all_level_labels[keep_indices]
            else:
                final_boxes = torch.empty((0, 6), device=device)
                final_scores = torch.empty((0,), device=device)
                final_labels = torch.empty((0,), dtype=torch.long, device=device)

            all_boxes.append(final_boxes)
            all_scores.append(final_scores)
            all_labels.append(final_labels)

        return {
            'boxes': all_boxes,
            'scores': all_scores,
            'labels': all_labels,
        }

    def _generate_grid(
        self,
        feat_size: List[int],
        image_shape: Tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """生成特征图每个位置对应的原图中心点坐标"""
        D, H, W = feat_size
        img_D, img_H, img_W = image_shape

        stride_z = img_D / D
        stride_y = img_H / H
        stride_x = img_W / W

        z_coords = (torch.arange(D, dtype=torch.float32, device=device) + 0.5) * stride_z
        y_coords = (torch.arange(H, dtype=torch.float32, device=device) + 0.5) * stride_y
        x_coords = (torch.arange(W, dtype=torch.float32, device=device) + 0.5) * stride_x

        grid_z, grid_y, grid_x = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

        return torch.stack([
            grid_z.flatten(),
            grid_y.flatten(),
            grid_x.flatten()
        ], dim=1)

    def _decode_boxes(
        self,
        center_points: torch.Tensor,
        reg_pred: torch.Tensor,
        image_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        YOLOv8风格框解码（推理版本）

        注意：与训练时的 _decode_boxes_yolov8 保持一致，使用 abs() 解码

        Args:
            center_points: [N, 3] 特征点中心坐标
            reg_pred: [N, 6] 预测的6个边界距离
            image_shape: 原图尺寸

        Returns:
            boxes: [N, 6] corner格式
        """
        # 使用abs()与训练时解码方式一致
        z1 = reg_pred[:, 0].abs()
        z2 = reg_pred[:, 3].abs()
        y1 = reg_pred[:, 1].abs()
        y2 = reg_pred[:, 4].abs()
        x1 = reg_pred[:, 2].abs()
        x2 = reg_pred[:, 5].abs()

        z_min = (center_points[:, 0] - z1).clamp(min=0, max=image_shape[0])
        z_max = (center_points[:, 0] + z2).clamp(min=0, max=image_shape[0])
        y_min = (center_points[:, 1] - y1).clamp(min=0, max=image_shape[1])
        y_max = (center_points[:, 1] + y2).clamp(min=0, max=image_shape[1])
        x_min = (center_points[:, 2] - x1).clamp(min=0, max=image_shape[2])
        x_max = (center_points[:, 2] + x2).clamp(min=0, max=image_shape[2])

        return torch.stack([z_min, y_min, x_min, z_max, y_max, x_max], dim=1)

    def _nms_3d(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """3D NMS"""
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)

        sorted_indices = torch.argsort(scores, descending=True)
        keep = []

        while sorted_indices.numel() > 0:
            if len(keep) >= self.max_detections:
                break

            current_idx = sorted_indices[0]
            keep.append(current_idx)

            if sorted_indices.numel() == 1:
                break

            remaining_indices = sorted_indices[1:]
            iou = self._compute_iou_3d(boxes[current_idx], boxes[remaining_indices])

            mask = iou < self.nms_iou_threshold
            sorted_indices = remaining_indices[mask]

        return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)

    def _compute_iou_3d(self, box1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """计算一个框与多个框的IoU"""
        if boxes2.numel() == 0:
            return torch.empty((0,), device=box1.device)

        box1 = box1.unsqueeze(0).expand(boxes2.shape[0], -1)

        inter_min = torch.max(box1[:, :3], boxes2[:, :3])
        inter_max = torch.min(box1[:, 3:], boxes2[:, 3:])
        inter_size = (inter_max - inter_min).clamp(min=0)
        inter_vol = inter_size[:, 0] * inter_size[:, 1] * inter_size[:, 2]

        vol1 = (box1[:, 3] - box1[:, 0]) * (box1[:, 4] - box1[:, 1]) * (box1[:, 5] - box1[:, 2])
        vol2 = (boxes2[:, 3] - boxes2[:, 0]) * (boxes2[:, 4] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 2])

        union_vol = vol1 + vol2 - inter_vol

        return inter_vol / (union_vol + 1e-6)


class MSDynUnetAnchorFreeDetector(nn.Module):
    """
    基于DynUNet的YOLOv8风格无锚框3D目标检测器

    继承自MSDynUnetDetector结构，替换为YOLOv8风格无锚框检测头
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        filters: List[int] = None,
        strides: List[List[int]] = None,
        kernel_size: List[List[int]] = None,
        upsample_kernel_size: List[List[int]] = None,
        feat_channels: int = 256,
        use_gn: bool = True,
        num_classes: int = 1,
        score_threshold: float = 0.25,
        nms_iou_threshold: float = 0.5,
        max_detections: int = 100,
        loss_type: str = 'diou',
        neg_iou_thr: float = 0.5,
        cls_weight: float = 1.0,
        bbox_weight: float = 2.0,
        max_pos_per_gt: int = 3,
        neg_sample_random_ratio: float = 1.0,
        neg_sample_hard_ratio: float = 0.01,
        use_dfl: bool = False,
        dfl_weight: float = 0.25,
        ciou_weight: float = 0.75,
        dfl_bins: int = 18,
        use_global_context: bool = True,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        assignment_mode: str = "classic",
        assign_use_pred_iou: bool = True,
        assign_one_level: bool = False,
        assignment_quality: str = "distance",
        assign_alpha: float = 1.0,
        assign_beta: float = 6.0,
        assign_expand_ratio: float = 1.5,
        assign_ignore_expand_ratio: float = 1.1,
        assign_soft_target_scores: bool = False,
        dfl_range_low: float = 0.0,
        dfl_range_high: float = 48.0,
        loss_normalization: str = "legacy",
        neg_cls_weight: float = 1.0,
        pos_cls_weight: float = 1.0,
        bbox_loss_norm: str = "num_pos",
        use_hard_negative_weight: bool = False,
        hard_negative_cls_weight: float = 1.5,
        hard_negative_score_threshold: float = 0.3,
    ):
        """
        Args:
            spatial_dims: 空间维度数
            in_channels: 输入通道数
            filters: 各阶段通道数列表
            strides: 各阶段步长列表
            kernel_size: 卷积核大小
            upsample_kernel_size: 上采样卷积核大小
            feat_channels: 检测头特征通道数
            use_gn: 是否使用GroupNorm
            num_classes: 类别数
            score_threshold: 置信度阈值
            nms_iou_threshold: NMS IoU阈值
            max_detections: 最大检测数
            loss_type: 损失类型 ('diou' 或 'ciou')
            cls_weight: 分类损失权重
            bbox_weight: 框回归损失权重
            max_pos_per_gt: 每个GT最多分配的正样本数量
            neg_sample_random_ratio: 随机采样负样本的比例
            neg_sample_hard_ratio: 困难负样本挖掘的比例
            use_dfl: 是否使用DFL Loss
            dfl_weight: DFL损失权重
            ciou_weight: CIoU损失权重
            dfl_bins: DFL的bins数量
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        self.use_dfl = use_dfl
        self.dfl_bins = dfl_bins
        self.use_global_context = use_global_context

        # 默认配置
        if filters is None:
            filters = [64, 96, 128, 192]
        self.filters = filters

        if strides is None:
            strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        self.strides = strides

        if kernel_size is None:
            kernel_size = [[3, 3, 3]] * len(strides)
        self.kernel_size = kernel_size

        if upsample_kernel_size is None:
            upsample_kernel_size = [[2, 2, 2], [2, 2, 2]]
        self.upsample_kernel_size = upsample_kernel_size

        # 创建DynUNet backbone
        self.backbone = nn.ModuleDict({
            'input_block': self._create_input_block(),
            'downsamples': self._create_downsamples(),
            'bottleneck': self._create_bottleneck(),
            'upsamples': self._create_upsamples(),
        })

        # 解码器卷积层
        bottleneck_ch = filters[-1]  # 192
        skip_24_ch = filters[2]     # 128
        skip_48_ch = filters[1]     # 96
        up1_out_ch = 128
        up2_out_ch = 96

        self.decoder_convs = nn.ModuleDict({
            'up1': nn.Sequential(
                nn.Conv3d(bottleneck_ch + skip_24_ch, up1_out_ch, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(up1_out_ch, affine=True),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            ),
            'up2': nn.Sequential(
                nn.Conv3d(up1_out_ch + skip_48_ch, up2_out_ch, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(up2_out_ch, affine=True),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            ),
        })

        # 金字塔注意力融合模块（4阶段，对应编码器各层）
        if self.use_global_context:
            self.pyramid_attention_blocks = self._create_pyramid_attention_blocks()

        # 检测头输入通道数（固定值）
        detection_in_channels = [192, 128, 96]

        # YOLOv8风格无锚框检测头
        self.detection_head = AnchorFreeHead3D(
            in_channels_list=detection_in_channels,
            feat_channels=feat_channels,
            num_classes=num_classes,
            num_levels=3,
            use_gn=use_gn,
            use_dfl=use_dfl,
            dfl_bins=dfl_bins,
            dfl_range_low=dfl_range_low,
            dfl_range_high=dfl_range_high,
        )

        # YOLOv8风格损失函数
        self.loss_fn = AnchorFreeLoss3D(
            num_classes=num_classes,
            cls_weight=cls_weight,
            bbox_weight=bbox_weight,
            loss_type=loss_type,
            neg_iou_thr=neg_iou_thr,
            max_pos_per_gt=max_pos_per_gt,
            neg_sample_random_ratio=neg_sample_random_ratio,
            neg_sample_hard_ratio=neg_sample_hard_ratio,
            use_dfl=use_dfl,
            dfl_weight=dfl_weight,
            ciou_weight=ciou_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            assignment_mode=assignment_mode,
            assign_use_pred_iou=assign_use_pred_iou,
            assign_one_level=assign_one_level,
            assignment_quality=assignment_quality,
            assign_alpha=assign_alpha,
            assign_beta=assign_beta,
            assign_expand_ratio=assign_expand_ratio,
            assign_ignore_expand_ratio=assign_ignore_expand_ratio,
            assign_soft_target_scores=assign_soft_target_scores,
            dfl_range_low=dfl_range_low,
            dfl_range_high=dfl_range_high,
            loss_normalization=loss_normalization,
            neg_cls_weight=neg_cls_weight,
            pos_cls_weight=pos_cls_weight,
            bbox_loss_norm=bbox_loss_norm,
            use_hard_negative_weight=use_hard_negative_weight,
            hard_negative_cls_weight=hard_negative_cls_weight,
            hard_negative_score_threshold=hard_negative_score_threshold,
        )

        # 后处理
        self.post_process = AnchorFreePostProcess3D(
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
            num_classes=num_classes,
            dfl_range_low=dfl_range_low,
            dfl_range_high=dfl_range_high,
        )

    def _create_input_block(self):
        """创建输入块"""
        return nn.Sequential(
            nn.Conv3d(self.in_channels, self.filters[0], kernel_size=self.kernel_size[0],
                     stride=self.strides[0], padding=1, bias=False),
            nn.InstanceNorm3d(self.filters[0], affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def _create_downsamples(self):
        """创建下采样块"""
        layers = nn.ModuleList()
        for i in range(len(self.strides) - 1):
            layer = nn.Sequential(
                nn.Conv3d(self.filters[i], self.filters[i + 1],
                         kernel_size=self.kernel_size[i + 1],
                         stride=self.strides[i + 1], padding=1, bias=False),
                nn.InstanceNorm3d(self.filters[i + 1], affine=True),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            )
            layers.append(layer)
        return layers

    def _create_bottleneck(self):
        """创建瓶颈层"""
        return nn.Sequential(
            nn.Conv3d(self.filters[-2], self.filters[-1],
                     kernel_size=self.kernel_size[-1],
                     stride=self.strides[-1], padding=1, bias=False),
            nn.InstanceNorm3d(self.filters[-1], affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def _create_upsamples(self):
        """创建上采样块"""
        layers = nn.ModuleList()

        in_channels_list = self.filters[1:3][::-1]
        out_channels_list = self.filters[0:2][::-1]
        kernel_list = self.kernel_size[1:3][::-1]
        stride_list = self.strides[1:3][::-1]
        upsample_kernel_list = self.upsample_kernel_size[::-1]

        for i in range(len(in_channels_list)):
            layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                nn.Conv3d(in_channels_list[i], out_channels_list[i],
                         kernel_size=kernel_list[i], padding=1, bias=False),
                nn.InstanceNorm3d(out_channels_list[i], affine=True),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            )
            layers.append(layer)

        return layers

    def _create_pyramid_attention_blocks(self):
        """创建金字塔注意力融合模块"""
        pyramid_attention_blocks = nn.ModuleDict()

        global_context_channels = [32, 64, 96, 128]

        for stage_idx in range(4):
            main_ch = self.filters[stage_idx]
            context_ch = global_context_channels[stage_idx]

            if context_ch != main_ch:
                proj = nn.Conv3d(context_ch, main_ch, kernel_size=1)
            else:
                proj = nn.Identity()

            pyramid_attention_blocks[f"proj_{stage_idx}"] = proj

            attention = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(main_ch, main_ch // 8, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(main_ch // 8, main_ch, kernel_size=1),
                nn.Sigmoid()
            )
            pyramid_attention_blocks[f"attention_{stage_idx}"] = attention

        return pyramid_attention_blocks

    def _fuse_with_pyramid(self, main_feat: torch.Tensor, context_feat: torch.Tensor,
                           stage_idx: int) -> torch.Tensor:
        """使用通道注意力融合主特征和上下文特征"""
        proj = self.pyramid_attention_blocks[f"proj_{stage_idx}"]
        attention = self.pyramid_attention_blocks[f"attention_{stage_idx}"]

        if context_feat.shape[2:] != main_feat.shape[2:]:
            context_feat = F.interpolate(
                context_feat, size=main_feat.shape[2:],
                mode='trilinear', align_corners=True
            )

        aligned_context = proj(context_feat)
        combined = main_feat + aligned_context
        attention_weights = attention(combined)
        fused = main_feat * attention_weights + aligned_context

        return fused

    def _extract_features(self, x: torch.Tensor, global_context: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        """提取特征金字塔

        Args:
            x: 输入影像 [B, C, D, H, W]
            global_context: 全局上下文网络输出的四阶段特征列表 (可选)
        """
        use_pyramid_fusion = global_context is not None and len(global_context) == 4

        x = self.backbone.input_block(x)

        # 阶段0融合
        if use_pyramid_fusion:
            x = self._fuse_with_pyramid(x, global_context[0], 0)

        encoder_outputs = [x]

        for down_idx, downsample in enumerate(self.backbone.downsamples):
            x = downsample(x)

            if use_pyramid_fusion:
                x = self._fuse_with_pyramid(x, global_context[down_idx + 1], down_idx + 1)

            encoder_outputs.append(x)

        bottleneck = encoder_outputs[-1]
        skip_24 = encoder_outputs[2]
        skip_48 = encoder_outputs[1]

        decoder_features = []
        x = bottleneck

        # Up1: 12³ → 24³
        x = F.interpolate(x, scale_factor=2, mode='trilinear')
        x = torch.cat([x, skip_24], dim=1)
        x = self.decoder_convs['up1'](x)
        decoder_features.append(x)

        # Up2: 24³ → 48³
        x = F.interpolate(x, scale_factor=2, mode='trilinear')
        x = torch.cat([x, skip_48], dim=1)
        x = self.decoder_convs['up2'](x)
        decoder_features.append(x)

        # 构建检测头输入: [Level0(12³,192ch), Level1(24³,128ch), Level2(48³,96ch)]
        detection_features = [
            bottleneck,
            decoder_features[0],
            decoder_features[1],
        ]

        return detection_features

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[List[Dict]] = None,
        global_context: Optional[List[torch.Tensor]] = None,
        debug: bool = True,
    ) -> Dict:
        """
        前向传播

        Args:
            x: 输入影像 [B, C, D, H, W]
            targets: 训练时的GT标注 (可选)
            global_context: 全局上下文网络输出的四阶段特征列表 (可选)
            debug: 是否打印调试信息

        Returns:
            训练模式: 损失字典
            推理模式: 检测结果字典
        """
        image_shape = x.shape[2:]

        # 提取特征（支持金字塔注意力融合）
        features = self._extract_features(x, global_context=global_context)

        # 获取特征图尺寸
        feature_sizes = [f.shape[2:] for f in features]

        # 检测头前向
        predictions = self.detection_head(features)

        # 训练模式
        if self.training and targets is not None:
            losses = self.loss_fn(predictions, targets, feature_sizes, image_shape, debug=debug)
            return losses

        # 推理模式
        results = self.post_process(predictions, feature_sizes, image_shape)

        return results


def build_anchor_free_detector(config: dict) -> MSDynUnetAnchorFreeDetector:
    """
    从配置构建YOLOv8风格无锚框检测器

    Args:
        config: 配置字典

    Returns:
        MSDynUnetAnchorFreeDetector实例
    """
    model_config = config.get('model', {})

    spatial_dims = model_config.get('spatial_dims', 3)
    in_channels = model_config.get('in_channels', 1)
    filters = model_config.get('filters', [64, 96, 128, 192])
    strides = model_config.get('strides', [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]])
    kernel_size = model_config.get('kernel_size', [[3, 3, 3]] * len(strides))
    upsample_kernel_size = [[2, 2, 2], [2, 2, 2]]

    detection_config = model_config.get('detection', {})
    feat_channels = detection_config.get('feat_channels', 256)
    use_gn = detection_config.get('use_gn', True)
    num_classes = detection_config.get('num_classes', 1)
    score_threshold = detection_config.get('score_threshold', 0.25)
    nms_iou_threshold = detection_config.get('nms_iou_threshold', 0.5)
    max_detections = detection_config.get('max_detections', 100)
    loss_type = detection_config.get('loss_type', 'diou')
    neg_iou_thr = detection_config.get('neg_iou_thr', 0.5)
    cls_weight = detection_config.get('cls_weight', 1.0)
    bbox_weight = detection_config.get('bbox_weight', 2.0)
    max_pos_per_gt = detection_config.get('max_pos_per_gt', 3)
    neg_sample_random_ratio = detection_config.get('neg_sample_random_ratio', 1.0)
    neg_sample_hard_ratio = detection_config.get('neg_sample_hard_ratio', 0.01)

    # DFL 配置
    use_dfl = detection_config.get('use_dfl', False)
    dfl_weight = detection_config.get('dfl_weight', 0.25)
    ciou_weight = detection_config.get('ciou_weight', 0.75)
    dfl_bins = detection_config.get('dfl_bins', 18)

    use_global_context = detection_config.get('use_global_context', True)

    # Focal Loss 配置
    focal_alpha = detection_config.get('focal_alpha', 0.75)
    focal_gamma = detection_config.get('focal_gamma', 2.0)
    assignment_mode = detection_config.get('assignment_mode', 'classic')
    assign_use_pred_iou = detection_config.get('assign_use_pred_iou', True)
    assign_one_level = detection_config.get('assign_one_level', False)
    assignment_quality = detection_config.get('assignment_quality', 'distance')
    assign_alpha = detection_config.get('assign_alpha', 1.0)
    assign_beta = detection_config.get('assign_beta', 6.0)
    assign_expand_ratio = detection_config.get('assign_expand_ratio', 1.5)
    assign_ignore_expand_ratio = detection_config.get('assign_ignore_expand_ratio', 1.1)
    assign_soft_target_scores = detection_config.get('assign_soft_target_scores', False)
    dfl_range_low = detection_config.get('dfl_range_low', 0.0)
    dfl_range_high = detection_config.get('dfl_range_high', 48.0)
    loss_normalization = detection_config.get('loss_normalization', 'legacy')
    neg_cls_weight = detection_config.get('neg_cls_weight', 1.0)
    pos_cls_weight = detection_config.get('pos_cls_weight', 1.0)
    bbox_loss_norm = detection_config.get('bbox_loss_norm', 'num_pos')
    use_hard_negative_weight = detection_config.get('use_hard_negative_weight', False)
    hard_negative_cls_weight = detection_config.get('hard_negative_cls_weight', 1.5)
    hard_negative_score_threshold = detection_config.get('hard_negative_score_threshold', 0.3)

    return MSDynUnetAnchorFreeDetector(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        filters=filters,
        strides=strides,
        kernel_size=kernel_size,
        upsample_kernel_size=upsample_kernel_size,
        feat_channels=feat_channels,
        use_gn=use_gn,
        num_classes=num_classes,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
        loss_type=loss_type,
        neg_iou_thr=neg_iou_thr,
        cls_weight=cls_weight,
        bbox_weight=bbox_weight,
        max_pos_per_gt=max_pos_per_gt,
        neg_sample_random_ratio=neg_sample_random_ratio,
        neg_sample_hard_ratio=neg_sample_hard_ratio,
        use_dfl=use_dfl,
        dfl_weight=dfl_weight,
        ciou_weight=ciou_weight,
        dfl_bins=dfl_bins,
        use_global_context=use_global_context,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        assignment_mode=assignment_mode,
        assign_use_pred_iou=assign_use_pred_iou,
        assign_one_level=assign_one_level,
        assignment_quality=assignment_quality,
        assign_alpha=assign_alpha,
        assign_beta=assign_beta,
        assign_expand_ratio=assign_expand_ratio,
        assign_ignore_expand_ratio=assign_ignore_expand_ratio,
        assign_soft_target_scores=assign_soft_target_scores,
        dfl_range_low=dfl_range_low,
        dfl_range_high=dfl_range_high,
        loss_normalization=loss_normalization,
        neg_cls_weight=neg_cls_weight,
        pos_cls_weight=pos_cls_weight,
        bbox_loss_norm=bbox_loss_norm,
        use_hard_negative_weight=use_hard_negative_weight,
        hard_negative_cls_weight=hard_negative_cls_weight,
        hard_negative_score_threshold=hard_negative_score_threshold,
    )
