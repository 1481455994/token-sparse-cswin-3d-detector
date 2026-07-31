# Token-Sparse CSWin DynUNet Detector

## Install

```bash
pip install -e .
```

The runtime dependencies are PyTorch, MONAI, and NumPy. Install a PyTorch build
compatible with your CUDA environment before installing this package if GPU
inference is required.

## Instantiate the paper model

```python
import torch
from tscswin_detector import build_token_sparse_cswin_adapter_dynunet_anchor_free_detector

config = {
    "model": {
        "in_channels": 1,
        "out_channels": 1,
        "spatial_dims": 3,
        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        "kernel_size": [[3, 3, 3]] * 4,
        "upsample_kernel_size": [[2, 2, 2]] * 3,
        "filters": [64, 96, 128, 192],
        "cswin_adapter": {
            "enabled": True,
            "stages": ["encoder_24", "bottleneck_12"],
            "input_size": 96,
            "num_heads": 8,
            "depths": {"encoder_24": 1, "bottleneck_12": 1},
            "mlp_ratio": 2.0,
            "axis_head_splits": [[2, 3, 3], [3, 2, 3], [3, 3, 2]],
            "stripe_window_D": 2, "stripe_window_H": 2, "stripe_window_W": 2,
            "use_lesion_gate": True, "lesion_gate_hidden_channels": 8,
        },
        "sparse_cswin_adapter": {
            "enabled": True, "stages": ["encoder_48"], "input_size": 96,
            "depth": 1, "num_heads": 8,
            "axis_head_splits": [[2, 3, 3], [3, 2, 3], [3, 3, 2]],
            "stripe_window_D": 2, "stripe_window_H": 2, "stripe_window_W": 2,
            "query_top_k_per_window": 16, "use_soft_gate": True,
        },
        "skip_se": {"enabled": True, "stages": ["encoder_24", "encoder_48"]},
        "detection": {
            "num_classes": 1, "feat_channels": 192, "loss_type": "ciou",
            "score_threshold": 0.3, "nms_iou_threshold": 0.3,
            "use_gn": True, "use_dfl": False,
        },
    }
}
model = build_token_sparse_cswin_adapter_dynunet_anchor_free_detector(config)
model.load_state_dict(torch.load("checkpoint.pt", map_location="cpu")["model_state_dict"])
model.eval()

with torch.inference_mode():
    output = model(torch.randn(1, 1, 96, 96, 96))
```

The model expects a five-dimensional tensor `[B, C, D, H, W]`. The CSWin
adapters in the paper configuration are fixed to 96-cubed input patches.
Inference returns a dictionary with decoded detection tensors and `gate_maps`.
For training, call `model(images, targets=targets)` in training mode; `targets`
is a list of dictionaries consumed by `AnchorFreeLoss3D`.

## Repository layout

```text
src/tscswin_detector/
  detector.py          # Paper model and config builder
  cswin_detector.py    # DynUNet-CSWin detector base
  dynunet.py           # DynUNet backbone
  anchor_free_head.py  # Detection head, loss, and post-processing
  mssm/                # Dense and token-sparse 3D CSWin blocks
  losses/              # Internal detection losses
```

## License and third-party notice

This package is MIT-licensed. `dynunet.py` is derived from MONAI and retains
its Apache-2.0 header; see `NOTICE`.
