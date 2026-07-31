"""
目标检测损失函数模块
包括：Focal Loss, GIoU Loss, Smooth L1 Loss, 组合检测损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection
    论文: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """
        Args:
            alpha: Weighting factor in range (0,1) to balance positive vs negative examples
            gamma: Exponent of the modulating factor (1 - p_t)^gamma
            reduction: Specifies the reduction to apply to the output
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 预测的概率值 [N, ...]
            targets: 标签 [N, ...]
        
        Returns:
            loss: Focal loss值
        """
        # 计算二元交叉熵
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算pt
        p = torch.sigmoid(inputs)
        pt = p * targets + (1 - p) * (1 - targets)
        
        # 计算focal factor
        focal_factor = (1 - pt) ** self.gamma
        
        # 应用alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_factor * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss (也称为 Huber Loss)
    用于边界框回归
    """
    
    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = 'mean',
    ):
        """
        Args:
            beta: 过渡点，超出该范围使用L2损失
            reduction: 损失 reduction方式
        """
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 预测值 [N, 6] (dz, dy, dx, dd, dh, dw)
            targets: 目标值 [N, 6]
        
        Returns:
            loss: Smooth L1 loss值
        """
        diff = torch.abs(inputs - targets)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss
    论文: https://arxiv.org/abs/1902.09630
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 损失 reduction方式
        """
        super(GIoULoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes: 预测的边界框 [N, 6] (z_center, y_center, x_center, d, h, w)
            targets: 目标边界框 [N, 6]
        
        Returns:
            loss: GIoU loss值
        """
        # 转换为角点格式
        pred_corners = self._box_to_corners(pred_boxes)  # [N, 8, 3]
        target_corners = self._box_to_corners(target_boxes)  # [N, 8, 3]
        
        # 计算各自的边界
        pred_min, pred_max = pred_corners.min(dim=1)[0], pred_corners.max(dim=1)[0]
        target_min, target_max = target_corners.min(dim=1)[0], target_corners.max(dim=1)[0]
        
        # 计算交集
        inter_min = torch.max(pred_min, target_min)
        inter_max = torch.min(pred_max, target_max)
        inter = (inter_max - inter_min).clamp(min=0)
        inter_volume = inter[:, 0] * inter[:, 1] * inter[:, 2]
        
        # 计算各自的体积
        pred_volume = (pred_max - pred_min)[:, 0] * (pred_max - pred_min)[:, 1] * (pred_max - pred_min)[:, 2]
        target_volume = (target_max - target_min)[:, 0] * (target_max - target_min)[:, 1] * (target_max - target_min)[:, 2]
        
        # 计算并集
        union_volume = pred_volume + target_volume - inter_volume
        
        # 计算IoU
        iou = inter_volume / (union_volume + 1e-7)
        
        # 计算外接框
        enclose_min = torch.min(pred_min, target_min)
        enclose_max = torch.max(pred_max, target_max)
        enclose_volume = (enclose_max - enclose_min)[:, 0] * (enclose_max - enclose_min)[:, 1] * (enclose_max - enclose_min)[:, 2]
        
        # 计算GIoU
        giou = iou - (enclose_volume - union_volume) / (enclose_volume + 1e-7)
        
        # GIoU loss = 1 - GIoU
        loss = 1 - giou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def _box_to_corners(self, boxes: torch.Tensor) -> torch.Tensor:
        """将中心格式转换为角点格式"""
        z, y, x, d, h, w = boxes.unbind(-1)
        
        z_corners = torch.stack([z - d/2, z - d/2, z - d/2, z - d/2,
                                 z + d/2, z + d/2, z + d/2, z + d/2], dim=-1)
        y_corners = torch.stack([y - h/2, y - h/2, y + h/2, y + h/2,
                                 y - h/2, y - h/2, y + h/2, y + h/2], dim=-1)
        x_corners = torch.stack([x - w/2, x + w/2, x - w/2, x + w/2,
                                 x - w/2, x + w/2, x - w/2, x + w/2], dim=-1)
        
        corners = torch.stack([z_corners, y_corners, x_corners], dim=-1)
        return corners


class DIoULoss(nn.Module):
    """
    Distance IoU Loss
    论文: https://arxiv.org/abs/1911.08287
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(DIoULoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """计算DIoU Loss"""
        # 提取中心点和尺寸
        pred_z, pred_y, pred_x = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2]
        pred_d, pred_h, pred_w = pred_boxes[:, 3], pred_boxes[:, 4], pred_boxes[:, 5]
        
        target_z, target_y, target_x = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2]
        target_d, target_h, target_w = target_boxes[:, 3], target_boxes[:, 4], target_boxes[:, 5]
        
        # 计算中心距离
        center_dist_sq = (pred_z - target_z) ** 2 + (pred_y - target_y) ** 2 + (pred_x - target_x) ** 2
        
        # 计算外接框对角线距离
        pred_radius = torch.sqrt(pred_d**2 + pred_h**2 + pred_w**2) / 2
        target_radius = torch.sqrt(target_d**2 + target_h**2 + target_w**2) / 2
        diagonal_dist_sq = (pred_radius + target_radius) ** 2
        
        # 计算IoU (简化版)
        pred_volume = pred_d * pred_h * pred_w
        target_volume = target_d * target_h * target_w
        
        # 计算交集
        inter_d = torch.min(pred_d, target_d)
        inter_h = torch.min(pred_h, target_h)
        inter_w = torch.min(pred_w, target_w)
        inter_volume = inter_d * inter_h * inter_w
        
        union_volume = pred_volume + target_volume - inter_volume
        iou = inter_volume / (union_volume + 1e-7)
        
        # DIoU = IoU - center_dist_sq / diagonal_dist_sq
        diou = iou - center_dist_sq / (diagonal_dist_sq + 1e-7)
        
        loss = 1 - diou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DetectionLoss(nn.Module):
    """
    组合检测损失函数
    包括分类损失 (Focal Loss) 和 边界框回归损失 (GIoU + SmoothL1)
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        bbox_loss_type: str = 'giou',  # 'giou', 'diou', 'smoothl1'
        cls_loss_weight: float = 1.0,
        bbox_loss_weight: float = 2.0,
    ):
        """
        Args:
            num_classes: 类别数 (包括背景)
            focal_alpha: Focal Loss alpha参数
            focal_gamma: Focal Loss gamma参数
            bbox_loss_type: 边界框损失类型
            cls_loss_weight: 分类损失权重
            bbox_loss_weight: 边界框损失权重
        """
        super(DetectionLoss, self).__init__()
        
        self.num_classes = num_classes
        self.cls_loss_weight = cls_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        
        # 分类损失
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        # 边界框损失
        if bbox_loss_type == 'giou':
            self.bbox_loss = GIoULoss()
        elif bbox_loss_type == 'diou':
            self.bbox_loss = DIoULoss()
        else:
            self.bbox_loss = SmoothL1Loss(beta=1.0)
    
    def forward(
        self,
        rpn_cls_logits: Optional[torch.Tensor] = None,
        rpn_bbox_preds: Optional[torch.Tensor] = None,
        roi_cls_logits: Optional[torch.Tensor] = None,
        roi_bbox_preds: Optional[torch.Tensor] = None,
        rpn_labels: Optional[torch.Tensor] = None,
        rpn_bbox_targets: Optional[torch.Tensor] = None,
        roi_labels: Optional[torch.Tensor] = None,
        roi_bbox_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算检测损失
        
        Args:
            rpn_cls_logits: RPN分类logits [N, num_anchors]
            rpn_bbox_preds: RPN边界框预测 [N, num_anchors * 6]
            roi_cls_logits: ROI分类logits [N, num_proposals, num_classes]
            roi_bbox_preds: ROI边界框预测 [N, num_proposals, num_classes * 6]
            rpn_labels: RPN标签 [N, num_anchors]
            rpn_bbox_targets: RPN边界框目标 [N, num_anchors, 6]
            roi_labels: ROI标签 [N, num_proposals]
            roi_bbox_targets: ROI边界框目标 [N, num_proposals, 6]
        
        Returns:
            total_loss: 总损失
            loss_dict: 各分项损失字典
        """
        losses = {}
        # 注意：如果没有任何有效样本/分支被触发，仍要返回 Tensor
        # 否则上层会出现 `float/int has no attribute backward`。
        device = None
        for t in (rpn_cls_logits, rpn_bbox_preds, roi_cls_logits, roi_bbox_preds, rpn_labels, rpn_bbox_targets, roi_labels, roi_bbox_targets):
            if isinstance(t, torch.Tensor):
                device = t.device
                break
        if device is None:
            device = torch.device("cpu")
        total_loss = torch.zeros((), device=device)
        
        # RPN分类损失
        if rpn_cls_logits is not None and rpn_labels is not None:
            # Flatten
            rpn_cls_logits_flat = rpn_cls_logits.reshape(-1)
            rpn_labels_flat = rpn_labels.reshape(-1)
            
            # 只计算正负样本的损失
            valid_mask = rpn_labels_flat >= 0
            if valid_mask.sum() > 0:
                rpn_cls_loss = self.focal_loss(
                    rpn_cls_logits_flat[valid_mask],
                    rpn_labels_flat[valid_mask].float()
                )
                losses['rpn_cls_loss'] = rpn_cls_loss
                total_loss += rpn_cls_loss * self.cls_loss_weight
        
        # RPN边界框损失
        if rpn_bbox_preds is not None and rpn_bbox_targets is not None and rpn_labels is not None:
            # 只对正样本计算边界框损失
            pos_mask = rpn_labels > 0
            if pos_mask.sum() > 0:
                rpn_bbox_pred_pos = rpn_bbox_preds[pos_mask]
                rpn_bbox_target_pos = rpn_bbox_targets[pos_mask]
                rpn_bbox_loss = self.bbox_loss(rpn_bbox_pred_pos, rpn_bbox_target_pos)
                losses['rpn_bbox_loss'] = rpn_bbox_loss
                total_loss += rpn_bbox_loss * self.bbox_loss_weight
        
        # ROI分类损失
        if roi_cls_logits is not None and roi_labels is not None:
            roi_cls_logits_flat = roi_cls_logits.reshape(-1, self.num_classes)
            roi_labels_flat = roi_labels.reshape(-1)
            
            valid_mask = roi_labels_flat >= 0
            if valid_mask.sum() > 0:
                roi_cls_loss = F.cross_entropy(
                    roi_cls_logits_flat[valid_mask],
                    roi_labels_flat[valid_mask].long(),
                )
                losses['roi_cls_loss'] = roi_cls_loss
                total_loss += roi_cls_loss * self.cls_loss_weight
        
        # ROI边界框损失
        if roi_bbox_preds is not None and roi_bbox_targets is not None and roi_labels is not None:
            pos_mask = roi_labels > 0
            if pos_mask.sum() > 0:
                roi_bbox_pred_pos = roi_bbox_preds[pos_mask]
                roi_bbox_target_pos = roi_bbox_targets[pos_mask]
                roi_bbox_loss = self.bbox_loss(roi_bbox_pred_pos, roi_bbox_target_pos)
                losses['roi_bbox_loss'] = roi_bbox_loss
                total_loss += roi_bbox_loss * self.bbox_loss_weight
        
        return total_loss, losses


def build_detection_loss(config: dict) -> DetectionLoss:
    """
    从配置构建检测损失函数
    
    Args:
        config: 损失函数配置字典
    
    Returns:
        DetectionLoss实例
    """
    loss_config = config.get('loss', {})
    
    focal_config = loss_config.get('focal_loss', {})
    bbox_config = loss_config.get('bbox_loss', {})
    
    return DetectionLoss(
        num_classes=config.get('model', {}).get('detection', {}).get('num_classes', 1),
        focal_alpha=focal_config.get('alpha', 0.25),
        focal_gamma=focal_config.get('gamma', 2.0),
        bbox_loss_type=bbox_config.get('name', 'giou').lower(),
        cls_loss_weight=loss_config.get('cls_loss_weight', 1.0),
        bbox_loss_weight=loss_config.get('bbox_loss_weight', 2.0),
    )
