"""
Training script for balanced-ternary MNIST.

Usage
-----
poetry run python -m btmnist.train --epochs 8 --batch 128 --lr 3e-3 --delta 0.02 --device cuda \
    --use-triton-linear

Behavior
--------
- LeNet-ish CNN with BN **before** every ternary activation (critical).
- Optionally uses Triton autograd linear layers (forward popcount, backward STE).
- Trains end-to-end with STE; exports a ternary state for inference benchmarking.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from btmnist.quant import (
    TernaryConfig,
    TernaryConv2d,
    TernaryLinear,
    TernaryActivation,
    export_ternary_state,
)
from btmnist.kernel import TernaryLinearTriton


class BTNet(nn.Module):
    """LeNet-ish CNN with BN→Ternary activation blocks and optional Triton FCs."""

    def __init__(self, cfg: TernaryConfig, use_triton_linear: bool):
        super().__init__()
        # CONV1 (no activation inside conv; BN -> ternary act)
        self.conv1 = TernaryConv2d(1, 32, k=5, stride=1, padding=0, bias=False, cfg=cfg, act=False)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.act1 = TernaryActivation(cfg)
        self.pool1 = nn.MaxPool2d(2, 2)

        # CONV2
        self.conv2 = TernaryConv2d(32, 64, k=5, stride=1, padding=0, bias=False, cfg=cfg, act=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.act2 = TernaryActivation(cfg)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Flatten + BN before hitting the first FC ternary
        self.bn_flat = nn.BatchNorm1d(64 * 4 * 4, eps=1e-5, momentum=0.1)

        if use_triton_linear:
            self.fc1 = TernaryLinearTriton(64 * 4 * 4, 256, cfg=cfg, bias=True, act=True)
            self.fc2 = TernaryLinearTriton(256, 10, cfg=cfg, bias=True, act=False)  # logits
        else:
            self.fc1 = TernaryLinear(64 * 4 * 4, 256, bias=True, cfg=cfg, act=True)
            self.fc2 = TernaryLinear(256, 10, bias=True, cfg=cfg, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)
        x = self.bn_flat(x)  # BN before FC ternary helps a TON
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def get_data(batch: int) -> Tuple[DataLoader, DataLoader]:
    """Return MNIST train/test loaders."""
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    return DataLoader(train, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True), \
           DataLoader(test, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)


@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def train_main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8, help="Training epochs")
    p.add_argument("--batch", type=int, default=128, help="Batch size")
    p.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    p.add_argument("--delta", type=float, default=0.02, help="Dead-zone threshold for ternary projection")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--export", type=str, default="bt_mnist_export.pt", help="Path for exported ternary state")
    p.add_argument("--use-triton-linear", action="store_true", help="Use Triton autograd linear layers")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    cfg = TernaryConfig(delta=args.delta, learn_scales=True, per_channel=False, clip_activations=3.0)
    train_dl, test_dl = get_data(args.batch)
    model = BTNet(cfg, use_triton_linear=args.use_triton_linear).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        acc = evaluate(model, test_dl, device)
        print(f"Epoch {epoch:02d}: test acc {acc:.2f}%")

    export = export_ternary_state(model)
    torch.save(export, args.export)
    print(f"Exported ternary state to {args.export}")


if __name__ == "__main__":
    train_main()
