"""
Inference path using Triton GEMM for balanced-ternary linear layers.

Usage
-----
poetry run python -m btmnist.inference --export bt_mnist_export.pt --device cuda

Behavior
--------
- Loads exported ternary state.
- Builds the same model topology as training, but:
  - Convs run as PyTorch ternary layers (kept simple for now).
  - Final linear layer runs on Triton bit-plane GEMM.
- Reports test accuracy. Demonstrates kernel integration and correctness.

Notes
-----
- This still works unchanged with the new autograd training path.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from btmnist.quant import (
    TernaryConfig,
    TernaryConv2d,
    TernaryActivation,
    ternary,
)
from btmnist.kernel import pack_bitplanes, ternary_gemm_triton


class BTConvBackbone(nn.Module):
    """Convolutional front-end mirroring the training graph."""

    def __init__(self, cfg: TernaryConfig, export_state: dict):
        super().__init__()
        self.cfg = cfg
        self.conv1 = TernaryConv2d(1, 32, k=5, stride=1, padding=0, bias=False, cfg=cfg, act=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = TernaryConv2d(32, 64, k=5, stride=1, padding=0, bias=False, cfg=cfg, act=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.act = TernaryActivation(cfg)
        self._load_conv_weights(export_state)

    def _load_conv_weights(self, st: dict) -> None:
        with torch.no_grad():
            # conv1
            wq = st["conv1.weight_i8"].float()
            ap = st["conv1.alpha_p"].view(-1, 1, 1, 1)
            an = st["conv1.alpha_n"].view(-1, 1, 1, 1)
            w = torch.where(wq > 0, ap * wq, wq)
            w = torch.where(wq < 0, an * wq, w)
            self.conv1.w.latent.copy_(w)
            # conv2
            wq = st["conv2.weight_i8"].float()
            ap = st["conv2.alpha_p"].view(-1, 1, 1, 1)
            an = st["conv2.alpha_n"].view(-1, 1, 1, 1)
            w = torch.where(wq > 0, ap * wq, wq)
            w = torch.where(wq < 0, an * wq, w)
            self.conv2.w.latent.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.act(x)
        return x


@torch.no_grad()
def eval_with_triton(export_path: str, device: torch.device) -> float:
    """
    Load exported state and run evaluation using Triton GEMM for the final linear layers.
    """
    st = torch.load(export_path, map_location="cpu")
    cfg = TernaryConfig()
    backbone = BTConvBackbone(cfg, st).to(device)
    backbone.eval()

    # FC1
    fc1_wq = st["fc1.weight_i8"].to(device)
    fc1_ap = st["fc1.alpha_p"].to(device).reshape(-1)
    fc1_an = st["fc1.alpha_n"].to(device).reshape(-1)
    fc1_b = st.get("fc1.bias", torch.zeros(256)).to(device)

    # FC2
    fc2_wq = st["fc2.weight_i8"].to(device)
    fc2_ap = st["fc2.alpha_p"].to(device).reshape(-1)
    fc2_an = st["fc2.alpha_n"].to(device).reshape(-1)
    fc2_b = st.get("fc2.bias", torch.zeros(10)).to(device)

    tfm = transforms.Compose([transforms.ToTensor()])
    test = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    dl = DataLoader(test, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    correct = 0
    total = 0

    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        feats = backbone(x)                                 # [B, 64*4*4]
        feats_q = ternary(feats, cfg.delta).to(torch.int8)  # ensure ternary activations

        # FC1 via Triton GEMM
        A = feats_q.view(feats_q.shape[0], -1)              # [B, K]
        B = ternary(fc1_wq.float(), cfg.delta).to(torch.int8)   # [256, K]
        a_pos, a_neg = pack_bitplanes(A)
        b_pos, b_neg = pack_bitplanes(B)
        C1 = ternary_gemm_triton(a_pos, a_neg, b_pos, b_neg, fc1_ap, fc1_an, fc1_b)
        C1 = torch.clamp(C1, -cfg.clip_activations, cfg.clip_activations)
        C1_q = ternary(C1, cfg.delta).to(torch.int8)

        # FC2 via Triton GEMM
        A2 = C1_q                                         # [B, 256]
        B2 = ternary(fc2_wq.float(), cfg.delta).to(torch.int8)  # [10, 256]
        a2_pos, a2_neg = pack_bitplanes(A2)
        b2_pos, b2_neg = pack_bitplanes(B2)
        C2 = ternary_gemm_triton(a2_pos, a2_neg, b2_pos, b2_neg, fc2_ap, fc2_an, fc2_b)

        pred = C2.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    return 100.0 * correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export", type=str, required=True, help="Path to exported ternary state")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    acc = eval_with_triton(args.export, device)
    print(f"Triton inference accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()
