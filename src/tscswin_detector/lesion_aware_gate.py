from typing import Optional, Tuple

import torch
import torch.nn as nn


class LesionAwareGate3D(nn.Module):
    """
    Spatial lesion-aware gate for selective context fusion.

    The module predicts a [B, 1, D, H, W] gate map from local main features and
    context-enhanced residuals. It modulates where contextual information is
    allowed to alter the DynUNet feature map.
    """

    def __init__(self, channels: int, hidden_channels: int = 16, extra_channels: int = 0):
        super().__init__()
        self.channels = int(channels)
        self.extra_channels = int(extra_channels)
        if self.extra_channels < 0:
            raise ValueError(f"extra_channels must be non-negative, got {self.extra_channels}.")
        hidden_channels = max(4, min(hidden_channels, channels))
        self.gate = nn.Sequential(
            nn.Conv3d(channels * 2 + self.extra_channels, hidden_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=False),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        main_feat: torch.Tensor,
        context_residual: torch.Tensor,
        extra_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = [main_feat, context_residual]
        if extra_feat is not None:
            if self.extra_channels == 0:
                raise ValueError("extra_feat was provided, but this gate was built with extra_channels=0.")
            if extra_feat.shape[1] != self.extra_channels:
                raise ValueError(
                    f"Expected extra_feat with {self.extra_channels} channels, got {extra_feat.shape[1]}."
                )
            if extra_feat.shape[0] != main_feat.shape[0] or extra_feat.shape[2:] != main_feat.shape[2:]:
                raise ValueError(
                    f"Expected extra_feat shape [B, {self.extra_channels}, D, H, W] matching main_feat, "
                    f"got {tuple(extra_feat.shape)} for main_feat {tuple(main_feat.shape)}."
                )
            inputs.append(extra_feat)
        elif self.extra_channels > 0:
            raise ValueError("extra_feat is required when extra_channels > 0.")

        gate_map = self.gate(torch.cat(inputs, dim=1))
        fused = main_feat + gate_map * context_residual
        return fused, gate_map
