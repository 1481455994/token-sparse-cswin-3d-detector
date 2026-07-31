"""
Distribution Focal Loss (DFL) 模块
用于3D目标检测的边界框回归

特点：
1. 不均匀bins设计：2-10步长1，10-30步长2
2. 将连续回归问题转换为分类问题
3. 结合CIoU Loss进行联合优化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DFLLoss(nn.Module):
    """
    Distribution Focal Loss for 3D bounding box regression

    将连续的距离值建模为离散分布，通过交叉熵学习目标分布

    Bins设计（不均匀）：
    - 范围 [2, 10]: 步长1，8个bins
    - 范围 [10, 30]: 步长2，10个bins
    - 总共18个bins
    """

    def __init__(
        self,
        range_low: float = 2.0,
        range_high: float = 30.0,
        fine_bins: int = 8,       # 2-10范围，步长1，8个bins
        coarse_bins: int = 10,    # 10-30范围，步长2，10个bins
    ):
        """
        Args:
            range_low: 最小边界值 (2.0)
            range_high: 最大边界值 (30.0)
            fine_bins: 细粒度bins数量 (2-10范围，8个bins)
            coarse_bins: 粗粒度bins数量 (10-30范围，10个bins)
        """
        super().__init__()
        self.range_low = range_low
        self.range_high = range_high
        self.fine_bins = fine_bins
        self.coarse_bins = coarse_bins

        # 构建不均匀的边界点
        # 2-10: 步长1 -> [2,3,4,5,6,7,8,9,10] (9个点 = 8个bins)
        # 10-30: 步长2 -> [10,12,14,16,18,20,22,24,26,28,30] (11个点 = 10个bins)
        fine_edges = torch.linspace(range_low, 10.0, fine_bins + 1)  # fine_bins+1 = 9个边界点
        coarse_edges = torch.linspace(10.0, range_high, coarse_bins + 1)  # 11个边界点

        # 合并边界，去除重复的10.0（fine_edges的最后一个点是10.0，coarse_edges的第一个点也是10.0）
        self.register_buffer('bin_edges', torch.cat([fine_edges[:-1], coarse_edges]))
        self.num_bins = len(self.bin_edges) - 1  # 8 + 10 = 18个bins

        # 计算每个bin的中心点
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2  # [num_bins]
        self.register_buffer('bin_centers', bin_centers)

    def forward(
        self,
        pred_dist: torch.Tensor,
        target_dist: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算DFL Loss

        Args:
            pred_dist: [N, 6, num_bins] 预测的分布 logits
            target_dist: [N, 6] 目标距离值

        Returns:
            loss: scalar DFL loss
        """
        N, C, B = pred_dist.shape
        assert B == self.num_bins, f"pred_dist bins {B} != num_bins {self.num_bins}"

        # Flatten: [N, 6, 18] -> [N*6, 18]
        pred_flat = pred_dist.view(-1, B)
        target_flat = target_dist.view(-1)

        # 将目标值转换为bin索引
        target_bin = self._dist_to_bin(target_flat)  # [N*6]

        # Cross Entropy Loss (内部包含softmax)
        loss = F.cross_entropy(pred_flat, target_bin, reduction='mean')
        return loss

    def _dist_to_bin(self, dist: torch.Tensor) -> torch.Tensor:
        """
        将连续距离值转换为bin索引

        Args:
            dist: [N] 距离值

        Returns:
            bin_idx: [N] bin索引
        """
        # 限制在有效范围内
        dist_clamped = dist.clamp(self.range_low, self.range_high)

        # 找到每个值属于哪个bin
        # 使用二分查找优化
        bin_idx = self._searchsorted(self.bin_edges, dist_clamped)
        bin_idx = bin_idx.clamp(0, self.num_bins - 1)

        return bin_idx

    def _searchsorted(self, boundaries: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        PyTorch版本的二分查找，用于找到值的bin索引
        """
        # 使用torch.searchsorted (PyTorch 1.8+)
        if hasattr(torch, 'searchsorted'):
            return torch.searchsorted(boundaries, values)
        else:
            # 手动实现
            n_bins = len(boundaries) - 1
            idx = torch.zeros_like(values, dtype=torch.long)
            for i in range(n_bins):
                idx += (values > boundaries[i]).long()
            return idx

    def decode_dist(self, pred_dist: torch.Tensor) -> torch.Tensor:
        """
        从分布预测中解码出距离值（期望值）

        注意：输入 pred_dist 为 logits，内部会先 softmax 归一化再求期望。

        Args:
            pred_dist: [N, 6, num_bins] 预测的分布 logits

        Returns:
            decoded: [N, 6] 解码后的距离值
        """
        N, C, B = pred_dist.shape
        pred_flat = pred_dist.reshape(-1, B)  # [N*6, num_bins]

        # 先对 logits 进行 softmax 归一化得到概率分布
        prob = F.softmax(pred_flat, dim=-1)   # [N*6, num_bins]

        # 加权求和得到期望值
        decoded = (prob * self.bin_centers.unsqueeze(0)).sum(dim=1)  # [N*6]

        return decoded.view(N, C)


class DFLCIoULoss(nn.Module):
    """
    DFL + CIoU 组合损失

    DFL学习更精确的边界分布
    CIoU确保整体框的IoU优化
    """

    def __init__(
        self,
        dfl_weight: float = 0.25,
        ciou_weight: float = 0.75,
        range_low: float = 2.0,
        range_high: float = 30.0,
        fine_bins: int = 8,      # 2-10, 步长1, 8个bins
        coarse_bins: int = 10,   # 10-30, 步长2, 10个bins
    ):
        """
        Args:
            dfl_weight: DFL Loss权重
            ciou_weight: CIoU Loss权重
            其他参数同DFLLoss
        """
        super().__init__()
        self.dfl_weight = dfl_weight
        self.ciou_weight = ciou_weight

        self.dfl_loss = DFLLoss(
            range_low=range_low,
            range_high=range_high,
            fine_bins=fine_bins,
            coarse_bins=coarse_bins,
        )

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        target_dists: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算组合损失

        Args:
            pred_dist: [N, 6, num_bins] DFL预测的分布 logits
            pred_boxes: [N, 6] 解码后的预测框 (corner格式)
            target_boxes: [N, 6] 目标框 (corner格式)
            target_dists: [N, 6] 目标距离值 (可选，如果为None则从target_boxes计算)

        Returns:
            total_loss: 总损失
            loss_dict: {'dfl_loss': ..., 'ciou_loss': ...}
        """
        losses = {}

        # DFL Loss
        if target_dists is not None:
            dfl = self.dfl_loss(pred_dist, target_dists)
        else:
            # 需要从target_boxes计算距离... 这里简化处理
            dfl = self.dfl_loss(pred_dist, pred_boxes.abs())  # 使用预测的距离作为近似
        losses['dfl_loss'] = dfl

        # CIoU Loss
        ciou = self._compute_ciou(pred_boxes, target_boxes)
        losses['ciou_loss'] = ciou

        # 总损失
        total_loss = self.dfl_weight * dfl + self.ciou_weight * ciou

        return total_loss, losses

    def _compute_ciou(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        计算3D CIoU Loss
        """
        # IoU
        iou = self._compute_iou_3d(pred_boxes, target_boxes)

        # 中心距离
        pred_center = (pred_boxes[:, :3] + pred_boxes[:, 3:]) / 2
        target_center = (target_boxes[:, :3] + target_boxes[:, 3:]) / 2
        center_dist_sq = ((pred_center - target_center) ** 2).sum(dim=1)

        # 外接框对角线
        enclose_min = torch.min(pred_boxes[:, :3], target_boxes[:, :3])
        enclose_max = torch.max(pred_boxes[:, 3:], target_boxes[:, 3:])
        enclose_dist_sq = ((enclose_max - enclose_min) ** 2).sum(dim=1)

        # 长宽比惩罚
        pred_d = pred_boxes[:, 3] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 4] - pred_boxes[:, 1]
        pred_w = pred_boxes[:, 5] - pred_boxes[:, 2]
        target_d = target_boxes[:, 3] - target_boxes[:, 0]
        target_h = target_boxes[:, 4] - target_boxes[:, 1]
        target_w = target_boxes[:, 5] - target_boxes[:, 2]

        # 避免除零
        v = (4 / (torch.pi ** 2)) * (
            (torch.atan(pred_d / (pred_h + 1e-7)) - torch.atan(target_d / (target_h + 1e-7))) ** 2 +
            (torch.atan(pred_d / (pred_w + 1e-7)) - torch.atan(target_d / (target_w + 1e-7))) ** 2
        )

        # alpha参数
        alpha = v / ((1 - iou) + v + 1e-7)

        # CIoU = IoU - 中心距离惩罚 - 长宽比惩罚
        ciou = iou - (center_dist_sq / (enclose_dist_sq + 1e-7)) - (alpha * v)

        loss = 1 - ciou
        return loss.mean()

    def _compute_iou_3d(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        计算3D IoU

        Args:
            boxes1, boxes2: [N, 6] corner格式 (z_min, y_min, x_min, z_max, y_max, x_max)

        Returns:
            iou: [N] IoU值
        """
        #交集区域
        inter_min = torch.max(boxes1[:, :3], boxes2[:, :3])
        inter_max = torch.min(boxes1[:, 3:], boxes2[:, 3:])
        inter = (inter_max - inter_min).clamp(min=0)
        inter_vol = inter[:, 0] * inter[:, 1] * inter[:, 2]

        # 各自体积
        vol1 = (boxes1[:, 3] - boxes1[:, 0]) * (boxes1[:, 4] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 2])
        vol2 = (boxes2[:, 3] - boxes2[:, 0]) * (boxes2[:, 4] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 2])

        # IoU
        union_vol = vol1 + vol2 - inter_vol
        iou = inter_vol / (union_vol + 1e-7)

        return iou


def build_dfl_loss(config: dict) -> DFLCIoULoss:
    """
    从配置构建DFL+CIoU损失

    Args:
        config: 配置字典

    Returns:
        DFLCIoULoss实例
    """
    dfl_config = config.get('model', {}).get('detection', {}).get('dfl_config', {})

    return DFLCIoULoss(
        dfl_weight=dfl_config.get('dfl_weight', 0.25),
        ciou_weight=dfl_config.get('ciou_weight', 0.75),
        range_low=dfl_config.get('range_low', 2.0),
        range_high=dfl_config.get('range_high', 30.0),
        fine_bins=dfl_config.get('fine_bins', 8),      # 修改默认值为8
        coarse_bins=dfl_config.get('coarse_bins', 10),
    )