from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .dynunet import DynUNet
from .anchor_free_head import AnchorFreeHead3D, AnchorFreeLoss3D, AnchorFreePostProcess3D
from .lesion_aware_gate import LesionAwareGate3D

from .mssm.split_head import SplitHeadCSwinBlock3D


class CSWinResidualAdapter3D(nn.Module):
    """Residual CSWin adapter inserted into low-resolution DynUNet features."""

    def __init__(
        self,
        stage_name: str,
        channels: int,
        spatial_size: int,
        depth: int,
        num_heads: int,
        axis_head_splits: Sequence[Sequence[int]],
        mlp_ratio: float,
        stripe_window_D: int,
        stripe_window_H: int,
        stripe_window_W: int,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rates: Sequence[float],
        gamma_init: float,
        use_lesion_gate: bool,
        lesion_gate_hidden_channels: int,
        shift_window: bool = False,
        shift_size_D: int = 0,
        shift_size_H: int = 0,
        shift_size_W: int = 0,
        shift_alternate: bool = True,
        use_axis_aware_gate: bool = False,
        axis_desc_channels: int = 3,
        axis_desc_mode: str = "mean_var_range",
        fusion_mode: str = "residual_gate",
        fusion_conv_norm: bool = False,
        fusion_conv_act: bool = False,
    ):
        super().__init__()
        if depth < 0:
            raise ValueError(f"{stage_name} depth must be non-negative.")
        if len(axis_head_splits) == 0:
            raise ValueError("axis_head_splits must not be empty.")
        num_splits = len(axis_head_splits)
        total_blocks = depth * num_splits
        if len(drop_path_rates) != total_blocks:
            raise ValueError(
                f"drop_path_rates length ({len(drop_path_rates)}) must equal depth * num_splits "
                f"({depth} * {num_splits} = {total_blocks})."
            )

        self.stage_name = stage_name
        self.channels = channels
        self.spatial_size = spatial_size
        self.use_lesion_gate = bool(use_lesion_gate)
        self.shift_window = bool(shift_window)
        self.shift_size = (int(shift_size_D), int(shift_size_H), int(shift_size_W))
        self.shift_alternate = bool(shift_alternate)
        self.use_axis_aware_gate = bool(use_axis_aware_gate)
        self.axis_desc_channels = int(axis_desc_channels)
        self.axis_desc_mode = str(axis_desc_mode)
        self.fusion_mode = str(fusion_mode)
        if self.fusion_mode not in {"residual_gate", "residual_concat_conv"}:
            raise ValueError(
                f"Unsupported fusion_mode={fusion_mode!r}. "
                "Expected 'residual_gate' or 'residual_concat_conv'."
            )
        if self.use_axis_aware_gate and not self.use_lesion_gate:
            raise ValueError("use_axis_aware_gate=True requires use_lesion_gate=True.")
        if self.use_axis_aware_gate and self.axis_desc_channels != 3:
            raise ValueError(
                "The current SplitHead-CSWin axis descriptor produces 3 channels "
                f"(mean, variance, range), got axis_desc_channels={self.axis_desc_channels}."
            )
        self.shifted_block_indices = []
        self.blocks = nn.ModuleList()
        block_idx = 0
        for _ in range(depth):
            for split in axis_head_splits:
                axis_heads = tuple(int(v) for v in split)
                block_shift_enabled = self.shift_window and (
                    not self.shift_alternate or block_idx % 2 == 1
                )
                if block_shift_enabled:
                    self.shifted_block_indices.append(block_idx)
                self.blocks.append(
                    SplitHeadCSwinBlock3D(
                        dim=channels,
                        D_size=spatial_size,
                        H_size=spatial_size,
                        W_size=spatial_size,
                        num_heads=num_heads,
                        axis_heads=axis_heads,
                        stripe_window_D=stripe_window_D,
                        stripe_window_H=stripe_window_H,
                        stripe_window_W=stripe_window_W,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path_rates[block_idx],
                        gamma_init=gamma_init,
                        shift_enabled=block_shift_enabled,
                        shift_size_D=self.shift_size[0],
                        shift_size_H=self.shift_size[1],
                        shift_size_W=self.shift_size[2],
                        return_axis_desc=self.use_axis_aware_gate,
                        axis_desc_mode=self.axis_desc_mode,
                    )
                )
                block_idx += 1

        self.gate = (
            LesionAwareGate3D(
                channels=channels,
                hidden_channels=lesion_gate_hidden_channels,
                extra_channels=self.axis_desc_channels if self.use_axis_aware_gate else 0,
            )
            if self.use_lesion_gate
            else None
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        if self.fusion_mode == "residual_concat_conv":
            fusion_layers = [nn.Conv3d(channels * 2, channels, kernel_size=1, bias=True)]
            if fusion_conv_norm:
                fusion_layers.append(nn.InstanceNorm3d(channels, affine=True))
            if fusion_conv_act:
                fusion_layers.append(nn.GELU())
            self.fusion_conv = nn.Sequential(*fusion_layers)
        else:
            self.fusion_conv = None

    def forward(self, main_feat: torch.Tensor):
        context_feat = main_feat
        axis_desc_list = []
        for block in self.blocks:
            if self.use_axis_aware_gate:
                context_feat, axis_desc = block(context_feat)
                axis_desc_list.append(axis_desc)
            else:
                context_feat = block(context_feat)
        context_residual = context_feat - main_feat
        axis_desc = torch.stack(axis_desc_list, dim=0).mean(dim=0) if axis_desc_list else None

        if self.gate is not None:
            gated_feat, gate_map = self.gate(main_feat, context_residual, axis_desc)
            gated_residual = gated_feat - main_feat
        else:
            gate_map = None
            gated_residual = context_residual
        if self.fusion_conv is not None:
            fusion_input = torch.cat([main_feat, gated_residual], dim=1)
            fused_feat = main_feat + self.gamma * self.fusion_conv(fusion_input)
        else:
            fused_feat = main_feat + self.gamma * gated_residual
        return fused_feat, gate_map


class CSWinAdapterDynUNetAnchorFreeDetector(DynUNet):
    """
    DynUNet anchor-free detector with embedded split-head CSWin residual adapters.

    The adapters are inserted into low-resolution semantic features instead of
    building a separate context branch from the original CT volume.
    """

    valid_adapter_stages = ("encoder_48", "encoder_24", "bottleneck_12")

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: Sequence[Union[Sequence[int], int]] = ((3, 3, 3),) * 4,
        strides: Sequence[Union[Sequence[int], int]] = ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        upsample_kernel_size: Sequence[Union[Sequence[int], int]] = ((2, 2, 2), (2, 2, 2), (2, 2, 2)),
        filters: Optional[Sequence[int]] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        res_block: bool = False,
        trans_bias: bool = False,
        feat_channels: int = 256,
        use_gn: bool = True,
        num_classes: int = 1,
        score_threshold: float = 0.25,
        nms_iou_threshold: float = 0.5,
        max_detections: int = 100,
        loss_type: str = "diou",
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
        cswin_adapter_config: Optional[Dict] = None,
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            filters=filters,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
            deep_supervision=False,
            deep_supr_num=1,
            res_block=res_block,
            trans_bias=trans_bias,
        )

        if spatial_dims != 3:
            raise ValueError("CSWinAdapterDynUNetAnchorFreeDetector supports only 3D inputs.")
        if len(self.filters) < 4:
            raise ValueError("At least four DynUNet filter stages are required.")
        if len(self.upsamples) < 2:
            raise ValueError("At least two decoder stages are required for detection features.")

        self.adapters = self._create_adapters(cswin_adapter_config or {})
        self.last_gate_maps: Dict[str, torch.Tensor] = {}

        self.upsamples = nn.ModuleList(list(self.upsamples[:2]))
        self.skip_layers = nn.Identity()
        self.output_block = nn.Identity()
        if hasattr(self, "deep_supervision_heads"):
            self.deep_supervision_heads = nn.ModuleList()

        detection_in_channels = [self.filters[-1], self.filters[-2], self.filters[-3]]
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
        self.post_process = AnchorFreePostProcess3D(
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
            num_classes=num_classes,
            dfl_range_low=dfl_range_low,
            dfl_range_high=dfl_range_high,
        )

    def _create_adapters(self, config: Dict) -> nn.ModuleDict:
        adapters = nn.ModuleDict()
        if not config.get("enabled", True):
            return adapters

        stages = tuple(config.get("stages", ["encoder_24", "bottleneck_12"]))
        invalid_stages = [stage for stage in stages if stage not in self.valid_adapter_stages]
        if invalid_stages:
            raise ValueError(f"Unsupported cswin_adapter stages: {invalid_stages}.")

        depths = config.get("depths", {"encoder_24": 2, "bottleneck_12": 1})
        num_heads = int(config.get("num_heads", 8))
        axis_head_splits = self._resolve_axis_head_splits(config.get("axis_head_splits"), num_heads)
        enabled_depths = {stage: max(0, int(depths.get(stage, 0))) for stage in stages}
        total_blocks = sum(depth * len(axis_head_splits) for depth in enabled_depths.values())
        if total_blocks == 0:
            return adapters

        dpr = [x.item() for x in torch.linspace(0, float(config.get("drop_path_rate", 0.1)), total_blocks)]
        dpr_index = 0
        input_size = int(config.get("input_size", 96))
        stripe_window_D = int(config.get("stripe_window_D", 2))
        stripe_window_H = int(config.get("stripe_window_H", 2))
        stripe_window_W = int(config.get("stripe_window_W", 2))
        shift_window = bool(config.get("shift_window", config.get("use_shift_window", False)))
        shift_size_D = int(config.get("shift_size_D", max(0, stripe_window_D // 2)))
        shift_size_H = int(config.get("shift_size_H", max(0, stripe_window_H // 2)))
        shift_size_W = int(config.get("shift_size_W", max(0, stripe_window_W // 2)))
        shift_alternate = bool(config.get("shift_alternate", True))
        use_axis_aware_gate = bool(config.get("use_axis_aware_gate", False))
        axis_desc_channels = int(config.get("axis_desc_channels", 3))
        axis_desc_mode = str(config.get("axis_desc_mode", "mean_var_range"))
        fusion_mode = str(config.get("fusion_mode", "residual_gate"))
        fusion_conv_norm = bool(config.get("fusion_conv_norm", False))
        fusion_conv_act = bool(config.get("fusion_conv_act", False))

        stage_specs = {
            "encoder_48": (self.filters[1], input_size // 2),
            "encoder_24": (self.filters[2], input_size // 4),
            "bottleneck_12": (self.filters[3], input_size // 8),
        }
        for stage in self.valid_adapter_stages:
            depth = enabled_depths.get(stage, 0)
            if depth <= 0:
                continue
            channels, spatial_size = stage_specs[stage]
            stage_total_blocks = depth * len(axis_head_splits)
            adapters[stage] = CSWinResidualAdapter3D(
                stage_name=stage,
                channels=channels,
                spatial_size=spatial_size,
                depth=depth,
                num_heads=num_heads,
                axis_head_splits=axis_head_splits,
                mlp_ratio=float(config.get("mlp_ratio", 2.0)),
                stripe_window_D=stripe_window_D,
                stripe_window_H=stripe_window_H,
                stripe_window_W=stripe_window_W,
                qkv_bias=bool(config.get("qkv_bias", True)),
                drop_rate=float(config.get("drop_rate", 0.0)),
                attn_drop_rate=float(config.get("attn_drop_rate", 0.0)),
                drop_path_rates=dpr[dpr_index : dpr_index + stage_total_blocks],
                gamma_init=float(config.get("gamma_init", 0.0)),
                use_lesion_gate=bool(config.get("use_lesion_gate", True)),
                lesion_gate_hidden_channels=int(config.get("lesion_gate_hidden_channels", 8)),
                shift_window=shift_window,
                shift_size_D=shift_size_D,
                shift_size_H=shift_size_H,
                shift_size_W=shift_size_W,
                shift_alternate=shift_alternate,
                use_axis_aware_gate=use_axis_aware_gate,
                axis_desc_channels=axis_desc_channels,
                axis_desc_mode=axis_desc_mode,
                fusion_mode=fusion_mode,
                fusion_conv_norm=fusion_conv_norm,
                fusion_conv_act=fusion_conv_act,
            )
            dpr_index += stage_total_blocks
        return adapters

    @staticmethod
    def _resolve_axis_head_splits(axis_head_splits, num_heads: int):
        if axis_head_splits is None:
            if num_heads == 8:
                axis_head_splits = ((2, 3, 3), (3, 2, 3), (3, 3, 2))
            elif num_heads == 4:
                axis_head_splits = ((1, 1, 2), (1, 2, 1), (2, 1, 1))
            else:
                raise ValueError("axis_head_splits must be provided when num_heads is not 4 or 8.")
        if len(axis_head_splits) == 0:
            raise ValueError("axis_head_splits must not be empty.")

        resolved = []
        for split in axis_head_splits:
            split = tuple(int(v) for v in split)
            if len(split) != 3:
                raise ValueError(f"Each axis head split must contain three values, got {split}.")
            if any(v <= 0 for v in split):
                raise ValueError(f"Axis head split values must be positive, got {split}.")
            if sum(split) != num_heads:
                raise ValueError(f"Axis head split {split} must sum to num_heads ({num_heads}).")
            resolved.append(split)
        return tuple(resolved)

    def _apply_adapter(self, stage_name: str, x: torch.Tensor) -> torch.Tensor:
        if stage_name not in self.adapters:
            return x
        x, gate_map = self.adapters[stage_name](x)
        if gate_map is not None:
            self.last_gate_maps[stage_name] = gate_map
        return x

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

        up_24 = self.upsamples[0](bottleneck, encoder_outputs[-1])
        up_48 = self.upsamples[1](up_24, encoder_outputs[-2])
        return [bottleneck, up_24, up_48]

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[List[Dict]] = None,
        global_context: Optional[List[torch.Tensor]] = None,
        debug: bool = False,
    ) -> Dict:
        del global_context

        image_shape = x.shape[2:]
        features = self.extract_detection_features(x)
        feature_sizes = [f.shape[2:] for f in features]
        predictions = self.detection_head(features)

        if self.training and targets is not None:
            losses = self.loss_fn(predictions, targets, feature_sizes, image_shape, debug=debug)
            losses["gate_maps"] = self.last_gate_maps
            return losses

        results = self.post_process(predictions, feature_sizes, image_shape)
        results["gate_maps"] = self.last_gate_maps
        return results


def build_cswin_adapter_dynunet_anchor_free_detector(config: dict) -> CSWinAdapterDynUNetAnchorFreeDetector:
    model_config = config.get("model", {})
    detection_config = model_config.get("detection", {})
    adapter_config = model_config.get("cswin_adapter", {})

    strides = model_config.get("strides", [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]])
    kernel_size = model_config.get("kernel_size", [[3, 3, 3]] * len(strides))
    upsample_kernel_size = model_config.get("upsample_kernel_size")
    if upsample_kernel_size is None or len(upsample_kernel_size) != len(strides) - 1:
        upsample_kernel_size = list(strides[1:])

    return CSWinAdapterDynUNetAnchorFreeDetector(
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
        cswin_adapter_config=adapter_config,
    )
