"""
Quantization modules for balanced-ternary training and inference.

Design
------
- TernarySTE: straight-through estimator for ternary projection with dead-zone ±delta.
- TernaryActivation: clamps and projects activations to {-1, 0, +1}.
- TernaryParam: latent FP weights with TTQ-style positive/negative scales.
- TernaryConv2d, TernaryLinear: PyTorch-native layers used for training.
- Utilities to export a ternary int8 and scale representation for Triton inference.

Notes
-----
- Training path uses standard PyTorch layers and STE so autograd works as expected.
- Inference can use Triton GEMM for linear layers with packed bit-planes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TernaryConfig:
    """Hyperparameters for ternarization."""
    delta: float = 0.05
    learn_scales: bool = True
    per_channel: bool = False
    clip_activations: float = 2.5


class TernarySTE(torch.autograd.Function):
    """
    Straight-Through Estimator for ternary projection.

    Forward:
        q(x) ∈ {-1, 0, +1}, using dead-zone ±delta.
    Backward:
        Pass gradients within |x| ≤ 1, zero outside.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, delta: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.delta = delta
        q = torch.zeros_like(x)
        q = torch.where(x > delta, torch.ones_like(x), q)
        q = torch.where(x < -delta, -torch.ones_like(x), q)
        return q

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        x, = ctx.saved_tensors
        mask = (x.abs() <= 1.0).to(grad_output.dtype)
        return grad_output * mask, None



def ternary(x: torch.Tensor, delta: float) -> torch.Tensor:
    """Apply TernarySTE to tensor x."""
    _r = TernarySTE.apply(x, delta)
    if not isinstance(_r, torch.Tensor):
        raise ValueError(f"TernarySTE.apply did not return a tensor, got type {type(_r)} instead")
    return _r


class TernaryActivation(nn.Module):
    """Ternary activation: clamp then project to {-1, 0, +1}."""

    def __init__(self, cfg: TernaryConfig):
        super().__init__()
        self.delta = cfg.delta
        self.preclip = cfg.clip_activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -self.preclip, self.preclip)
        return ternary(x, self.delta)


class TernaryParam(nn.Module):
    """
    Latent FP parameter with TTQ-style scales.

    Exposes a ternary-projected weight with learned alpha_p, alpha_n.
    If per_channel=True for conv, scales are per-out-channel.
    """

    def __init__(self, shape, cfg: TernaryConfig, per_channel: bool):
        super().__init__()
        self.latent = nn.Parameter(torch.empty(*shape))
        self.delta = cfg.delta
        self.per_channel = per_channel
        if per_channel:
            oc = shape[0]
            self.alpha_p = nn.Parameter(torch.ones(oc))
            self.alpha_n = nn.Parameter(torch.ones(oc))
        else:
            self.alpha_p = nn.Parameter(torch.tensor(1.0))
            self.alpha_n = nn.Parameter(torch.tensor(1.0))
        nn.init.kaiming_uniform_(self.latent, a=2.0 ** 0.5)

    def forward(self) -> torch.Tensor:
        q = ternary(self.latent, self.delta)
        if self.per_channel:
            oc = q.shape[0]
            ap = self.alpha_p.view(oc, 1, 1, 1)
            an = self.alpha_n.view(oc, 1, 1, 1)
        else:
            ap = self.alpha_p
            an = self.alpha_n
        w = torch.where(q > 0, ap * q, q)
        w = torch.where(q < 0, an * q, w)
        return w


class TernaryConv2d(nn.Module):
    """Conv2d with ternary weights and optional ternary activation."""

    def __init__(self, in_ch, out_ch, k, stride=1, padding=0, bias=False, cfg: TernaryConfig | None = None, act: bool = True):
        super().__init__()
        assert cfg is not None
        self.stride = stride
        self.padding = padding
        self.w = TernaryParam((out_ch, in_ch, k, k), cfg, per_channel=cfg.per_channel)
        self.use_bias = bias
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.act = TernaryActivation(cfg) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv2d(x, self.w(), bias=self.bias, stride=self.stride, padding=self.padding)
        return self.act(y)


class TernaryLinear(nn.Module):
    """Linear with ternary weights and optional ternary activation."""

    def __init__(self, in_f, out_f, bias=True, cfg: TernaryConfig | None = None, act: bool = True):
        super().__init__()
        assert cfg is not None
        self.w = TernaryParam((out_f, in_f), cfg, per_channel=False)
        self.use_bias = bias
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        self.act = TernaryActivation(cfg) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.w(), self.bias)
        return self.act(y)


def export_ternary_state(model: nn.Module) -> Dict[str, Any]:
    """
    Produce an export dict with int8 ternary weights in {-1,0,+1} and float32 scales/biases.

    Returns
    -------
    dict mapping:
        <layer>.weight_i8 : int8 tensor in {-1,0,+1}
        <layer>.alpha_p   : float tensor (per-layer or per-channel)
        <layer>.alpha_n   : float tensor (per-layer or per-channel)
        <layer>.bias      : float tensor (optional)
        meta.layers       : ordered list of layer names
    """
    export: dict[str, torch.Tensor | list] = {"meta.layers": []}
    for name, module in model.named_modules():
        if isinstance(module, (TernaryConv2d, TernaryLinear)):
            if not isinstance(export["meta.layers"], list):
                raise ValueError("export['meta.layers'] should be a list")
            export["meta.layers"].append(name)
            wq = ternary(module.w.latent, module.w.delta).to(torch.int8).cpu()
            export[f"{name}.weight_i8"] = wq
            export[f"{name}.alpha_p"] = module.w.alpha_p.detach().float().cpu()
            export[f"{name}.alpha_n"] = module.w.alpha_n.detach().float().cpu()
            if module.use_bias:
                if module.bias is None:
                    raise ValueError(f"Module {name} has use_bias=True but bias is None")
                export[f"{name}.bias"] = module.bias.detach().float().cpu()
    return export
