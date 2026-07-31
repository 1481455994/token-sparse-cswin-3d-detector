from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .cswin_detector import CSWinAdapterDynUNetAnchorFreeDetector
from .mssm.token_sparse import TokenSparseStripeCSwinAdapter3D


class ResidualSE3D(nn.Module):
    """Channel attention for skip features with baseline-preserving residual fusion."""

    def __init__(self, channels: int, reduction: int = 8, gamma_init: float = 0.0):
        super().__init__()
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}.")
        hidden_channels = max(1, int(channels) // int(reduction))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_weights = self.excitation(self.pool(x))
        return x + self.gamma * x * channel_weights


class TokenSparseCSWinAdapterDynUNetAnchorFreeDetector(CSWinAdapterDynUNetAnchorFreeDetector):
    """
    DynUNet anchor-free detector with token-sparse CSWin at the 48-scale encoder.

    The encoder_48 adapter keeps every D/H/W stripe window active, but only
    selected high-gate tokens inside each window serve as queries. Keys/values
    remain dense inside the same stripe window. Dense CSWin adapters are still
    used for encoder_24 and bottleneck_12.
    """

    valid_sparse_adapter_stages = ("encoder_48",)

    def __init__(
        self,
        *args,
        cswin_adapter_config: Optional[Dict] = None,
        sparse_cswin_adapter_config: Optional[Dict] = None,
        skip_se_config: Optional[Dict] = None,
        **kwargs,
    ):
        dense_config = self._filter_dense_adapter_config(cswin_adapter_config or {})
        super().__init__(*args, cswin_adapter_config=dense_config, **kwargs)
        self.sparse_adapters = self._create_sparse_adapters(sparse_cswin_adapter_config or {})
        self.skip_se_adapters = self._create_skip_se_adapters(skip_se_config or {})

    @staticmethod
    def _filter_dense_adapter_config(config: Dict) -> Dict:
        dense_config = dict(config)
        stages = dense_config.get("stages", ["encoder_24", "bottleneck_12"])
        dense_config["stages"] = [stage for stage in stages if stage != "encoder_48"]

        depths = dict(dense_config.get("depths", {"encoder_24": 2, "bottleneck_12": 1}))
        depths.pop("encoder_48", None)
        dense_config["depths"] = depths
        return dense_config

    def _create_sparse_adapters(self, config: Dict) -> nn.ModuleDict:
        sparse_adapters = nn.ModuleDict()
        if not config or not config.get("enabled", True):
            return sparse_adapters

        stages = tuple(config.get("stages", ["encoder_48"]))
        invalid_stages = [stage for stage in stages if stage not in self.valid_sparse_adapter_stages]
        if invalid_stages:
            raise ValueError(f"Unsupported sparse_cswin_adapter stages: {invalid_stages}.")
        if "encoder_48" not in stages:
            return sparse_adapters

        input_size = int(config.get("input_size", 96))
        spatial_size = input_size // 2
        channels = self.filters[1]
        num_heads = int(config.get("num_heads", 8))
        axis_head_splits = self._resolve_axis_head_splits(config.get("axis_head_splits"), num_heads)

        depths_config = config.get("depths", {})
        if isinstance(depths_config, dict):
            depth = int(depths_config.get("encoder_48", config.get("depth", 1)))
        else:
            depth = int(config.get("depth", 1))
        if depth <= 0:
            return sparse_adapters

        total_blocks = depth * len(axis_head_splits)
        drop_path_rate = float(config.get("drop_path_rate", 0.0))
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        sparse_adapters["encoder_48"] = TokenSparseStripeCSwinAdapter3D(
            stage_name="encoder_48",
            channels=channels,
            spatial_size=spatial_size,
            query_top_k_per_window=int(config.get("query_top_k_per_window", 16)),
            min_gate_score=config.get("min_gate_score", config.get("gate_threshold", None)),
            depth=depth,
            num_heads=num_heads,
            axis_head_splits=axis_head_splits,
            mlp_ratio=float(config.get("mlp_ratio", 1.0)),
            stripe_window_D=int(config.get("stripe_window_D", 2)),
            stripe_window_H=int(config.get("stripe_window_H", 2)),
            stripe_window_W=int(config.get("stripe_window_W", 2)),
            qkv_bias=bool(config.get("qkv_bias", True)),
            drop_rate=float(config.get("drop_rate", 0.0)),
            attn_drop_rate=float(config.get("attn_drop_rate", 0.0)),
            drop_path_rates=drop_path_rates,
            gamma_init=float(config.get("gamma_init", 0.0)),
            use_soft_gate=bool(config.get("use_soft_gate", True)),
            gate_hidden_channels=config.get("gate_hidden_channels", None),
            fusion_mode=str(config.get("fusion_mode", "residual_gate")),
            fusion_conv_norm=bool(config.get("fusion_conv_norm", False)),
            fusion_conv_act=bool(config.get("fusion_conv_act", False)),
        )
        return sparse_adapters

    def _create_skip_se_adapters(self, config: Dict) -> nn.ModuleDict:
        skip_se_adapters = nn.ModuleDict()
        if not config or not config.get("enabled", False):
            return skip_se_adapters

        valid_stages = ("encoder_24", "encoder_48")
        stages = tuple(config.get("stages", valid_stages))
        invalid_stages = [stage for stage in stages if stage not in valid_stages]
        if invalid_stages:
            raise ValueError(f"Unsupported skip_se stages: {invalid_stages}.")

        stage_channels = {
            "encoder_24": self.filters[2],
            "encoder_48": self.filters[1],
        }
        reduction = int(config.get("reduction", 8))
        gamma_init = float(config.get("gamma_init", 0.0))
        for stage in stages:
            skip_se_adapters[stage] = ResidualSE3D(
                channels=stage_channels[stage],
                reduction=reduction,
                gamma_init=gamma_init,
            )
        return skip_se_adapters

    def _apply_adapter(self, stage_name: str, x):
        if stage_name in self.sparse_adapters:
            x, gate_map = self.sparse_adapters[stage_name](x)
            if gate_map is not None:
                self.last_gate_maps[stage_name] = gate_map
            return x
        return super()._apply_adapter(stage_name, x)

    def _apply_skip_se(self, stage_name: str, x: torch.Tensor) -> torch.Tensor:
        if stage_name not in self.skip_se_adapters:
            return x
        return self.skip_se_adapters[stage_name](x)

    def extract_detection_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        self.last_gate_maps = {}

        x = self.input_block(x)
        encoder_outputs = [x]

        for down_idx, downsample in enumerate(self.downsamples):
            x = downsample(x)
            stage_idx = down_idx + 1
            if stage_idx == 1:
                x = self._apply_adapter("encoder_48", x)
            if stage_idx == 2:
                x = self._apply_adapter("encoder_24", x)
            encoder_outputs.append(x)

        bottleneck = self.bottleneck(encoder_outputs[-1])
        bottleneck = self._apply_adapter("bottleneck_12", bottleneck)

        skip_24 = self._apply_skip_se("encoder_24", encoder_outputs[-1])
        up_24 = self.upsamples[0](bottleneck, skip_24)
        skip_48 = self._apply_skip_se("encoder_48", encoder_outputs[-2])
        up_48 = self.upsamples[1](up_24, skip_48)
        return [bottleneck, up_24, up_48]

    def set_sparse_query_top_k_per_window(self, query_top_k_per_window: int):
        for adapter in self.sparse_adapters.values():
            adapter.set_query_top_k_per_window(query_top_k_per_window)


def build_token_sparse_cswin_adapter_dynunet_anchor_free_detector(
    config: dict,
) -> TokenSparseCSWinAdapterDynUNetAnchorFreeDetector:
    model_config = config.get("model", {})
    detection_config = model_config.get("detection", {})
    dense_adapter_config = model_config.get("cswin_adapter", {})
    sparse_adapter_config = model_config.get("sparse_cswin_adapter", {})
    skip_se_config = model_config.get("skip_se", {})

    strides = model_config.get("strides", [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]])
    kernel_size = model_config.get("kernel_size", [[3, 3, 3]] * len(strides))
    upsample_kernel_size = model_config.get("upsample_kernel_size")
    if upsample_kernel_size is None or len(upsample_kernel_size) != len(strides) - 1:
        upsample_kernel_size = list(strides[1:])

    return TokenSparseCSWinAdapterDynUNetAnchorFreeDetector(
        spatial_dims=model_config.get("spatial_dims", 3),
        in_channels=model_config.get("in_channels", 1),
        out_channels=model_config.get("out_channels", 1),
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=upsample_kernel_size,
        filters=model_config.get("filters", [64, 96, 128, 192]),
        dropout=model_config.get("dropout", None),
        norm_name=model_config.get("norm_name", ("INSTANCE", {"affine": True})),
        act_name=model_config.get("act_name", ("leakyrelu", {"inplace": True, "negative_slope": 0.01})),
        res_block=model_config.get("res_block", False),
        trans_bias=model_config.get("trans_bias", False),
        feat_channels=detection_config.get("feat_channels", 256),
        use_gn=detection_config.get("use_gn", True),
        num_classes=detection_config.get("num_classes", 1),
        score_threshold=detection_config.get("score_threshold", 0.25),
        nms_iou_threshold=detection_config.get("nms_iou_threshold", 0.5),
        max_detections=detection_config.get("max_detections", 100),
        loss_type=detection_config.get("loss_type", "diou"),
        neg_iou_thr=detection_config.get("neg_iou_thr", 0.5),
        cls_weight=detection_config.get("cls_weight", 1.0),
        bbox_weight=detection_config.get("bbox_weight", 2.0),
        max_pos_per_gt=detection_config.get("max_pos_per_gt", 3),
        neg_sample_random_ratio=detection_config.get("neg_sample_random_ratio", 1.0),
        neg_sample_hard_ratio=detection_config.get("neg_sample_hard_ratio", 0.01),
        use_dfl=detection_config.get("use_dfl", False),
        dfl_weight=detection_config.get("dfl_weight", 0.25),
        ciou_weight=detection_config.get("ciou_weight", 0.75),
        dfl_bins=detection_config.get("dfl_bins", 18),
        focal_alpha=detection_config.get("focal_alpha", 0.75),
        focal_gamma=detection_config.get("focal_gamma", 2.0),
        assignment_mode=detection_config.get("assignment_mode", "classic"),
        assign_use_pred_iou=detection_config.get("assign_use_pred_iou", True),
        assign_one_level=detection_config.get("assign_one_level", False),
        assignment_quality=detection_config.get("assignment_quality", "distance"),
        assign_alpha=detection_config.get("assign_alpha", 1.0),
        assign_beta=detection_config.get("assign_beta", 6.0),
        assign_expand_ratio=detection_config.get("assign_expand_ratio", 1.5),
        assign_ignore_expand_ratio=detection_config.get("assign_ignore_expand_ratio", 1.1),
        assign_soft_target_scores=detection_config.get("assign_soft_target_scores", False),
        dfl_range_low=detection_config.get("dfl_range_low", 0.0),
        dfl_range_high=detection_config.get("dfl_range_high", 48.0),
        loss_normalization=detection_config.get("loss_normalization", "legacy"),
        neg_cls_weight=detection_config.get("neg_cls_weight", 1.0),
        pos_cls_weight=detection_config.get("pos_cls_weight", 1.0),
        bbox_loss_norm=detection_config.get("bbox_loss_norm", "num_pos"),
        use_hard_negative_weight=detection_config.get("use_hard_negative_weight", False),
        hard_negative_cls_weight=detection_config.get("hard_negative_cls_weight", 1.5),
        hard_negative_score_threshold=detection_config.get("hard_negative_score_threshold", 0.3),
        cswin_adapter_config=dense_adapter_config,
        sparse_cswin_adapter_config=sparse_adapter_config,
        skip_se_config=skip_se_config,
    )
