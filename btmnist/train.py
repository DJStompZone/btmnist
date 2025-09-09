"""
Training script for balanced-ternary MNIST with tqdm/logging/TensorBoard and 🤗 Hub pushes.

Features
--------
- TQDM progress bars with per-batch loss, EMA loss, LR, ETA.
- Clean logging with timestamps.
- Optional TensorBoard logging via --tb.
- Best-accuracy checkpoint/export; optionally push to Hugging Face Hub.
- BatchNorm before every ternary activation for stable ternary training.

Usage
-----
poetry run python -m btmnist.train --epochs 8 --batch 128 --lr 3e-3 --delta 0.02 \
    --device cuda --use-triton-linear --tb --push-to-hub --hub-repo djstompzone/btmnist
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from btmnist.logging_utils import make_logger, MovingAvg
from btmnist.quant import (
    TernaryConfig,
    TernaryConv2d,
    TernaryLinear,
    TernaryActivation,
    export_ternary_state,
)
from btmnist.kernel import TernaryLinearTriton
from btmnist.hf_hub import HubSpec, ensure_repo, push_many, dump_json


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def choose_device(pref: str = "cuda") -> torch.device:
    """
    Pick a device and print actionable diagnostics if CUDA/Triton are missing.
    """
    if pref.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA unavailable: CPU-only PyTorch or missing driver. Falling back to CPU.")
            return torch.device("cpu")
        # Optional: Triton driver check (won't crash if Triton missing)
        try:
            from triton.runtime import driver as _drv  # type: ignore
            _ = _drv.backends
        except Exception as e:
            print(f"Triton cannot see NVIDIA driver ({e}); kernel path will fail. "
                  f"Either fix driver install or run without --use-triton-linear.")
        return torch.device("cuda")
    return torch.device("cpu")


class BTNet(nn.Module):
    """LeNet-ish CNN with BN→Ternary activation blocks and optional Triton FCs."""

    def __init__(self, cfg: TernaryConfig, use_triton_linear: bool):
        super().__init__()
        # CONV1
        self.conv1 = TernaryConv2d(1, 32, k=5, stride=1, padding=0, bias=False, cfg=cfg, act=False)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.act1 = TernaryActivation(cfg)
        self.pool1 = nn.MaxPool2d(2, 2)

        # CONV2
        self.conv2 = TernaryConv2d(32, 64, k=5, stride=1, padding=0, bias=False, cfg=cfg, act=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.act2 = TernaryActivation(cfg)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Flatten + BN before FC
        self.bn_flat = nn.BatchNorm1d(64 * 4 * 4, eps=1e-5, momentum=0.1)

        if use_triton_linear:
            self.fc1 = TernaryLinearTriton(64 * 4 * 4, 256, cfg=cfg, bias=True, act=True)
            self.fc2 = TernaryLinearTriton(256, 10, cfg=cfg, bias=True, act=False)
        else:
            self.fc1 = TernaryLinear(64 * 4 * 4, 256, bias=True, cfg=cfg, act=True)
            self.fc2 = TernaryLinear(256, 10, bias=True, cfg=cfg, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x); x = self.bn1(x); x = self.act1(x); x = self.pool1(x)
        x = self.conv2(x); x = self.bn2(x); x = self.act2(x); x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.bn_flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def get_data(batch: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    """Return MNIST train/test loaders."""
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    return (
        DataLoader(train, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(test, batch_size=512, shuffle=False, num_workers=num_workers, pin_memory=True),
    )


@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
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
    p.add_argument("--seed", type=int, default=1337, help="Random seed")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--tb", action="store_true", help="Enable TensorBoard logging")

    p.add_argument("--push-to-hub", action="store_true", help="Push best checkpoint to Hugging Face Hub")
    p.add_argument("--hub-repo", type=str, default="", help="Target repo_id, e.g. djstompzone/btmnist")
    p.add_argument("--hub-private", action="store_true", help="Create repo as private if it doesn't exist")
    p.add_argument("--hub-branch", type=str, default=None, help="Branch to push to (default main)")
    p.add_argument("--hub-message", type=str, default="btmnist: update checkpoint", help="Commit message")
    args = p.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_LOG_LEVEL", "3")

    logger = make_logger()
    set_seed(args.seed)

    device = choose_device(args.device)
    logger.info(f"Device: {device.type}")
    logger.info(
        f"Args: epochs={args.epochs} batch={args.batch} lr={args.lr} delta={args.delta} "
        f"use_triton_linear={args.use_triton_linear} tb={args.tb}"
    )

    if device.type != "cuda" and args.use_triton_linear:
        logger.warning("CUDA not available; disabling Triton linear layers.")
        args.use_triton_linear = False

    cfg = TernaryConfig(delta=args.delta, learn_scales=True, per_channel=False, clip_activations=3.0)
    train_dl, test_dl = get_data(args.batch, args.num_workers)
    model = BTNet(cfg, use_triton_linear=args.use_triton_linear).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    writer: Optional["SummaryWriter"] = None
    global_step = 0
    if args.tb:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            logger.info("TensorBoard logging enabled (run `tensorboard --logdir runs` to view).")
        except Exception as e:
            logger.warning(f"TensorBoard init failed: {e}. Continuing without TB.")
            writer = None

    hub_spec: Optional[HubSpec] = None
    if args.push_to_hub:
        if not args.hub_repo:
            raise SystemExit("--push-to-hub requires --hub-repo owner/name")
        hub_spec = HubSpec(
            repo_id=args.hub_repo,
            repo_type="model",
            private=args.hub_private,
            branch=args.hub_branch,
            commit_message=args.hub_message,
            create_ok=True,
        )
        ensure_repo(hub_spec)
        logger.info(f"Hugging Face Hub ready → {args.hub_repo}"
                    f"{' (private)' if args.hub_private else ''}"
                    f"{f' [branch {args.hub_branch}]' if args.hub_branch else ''}")

    best_acc = 0.0
    ema = MovingAvg(beta=0.98)
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch:02d}/{args.epochs:02d}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            lr = opt.param_groups[0]["lr"]
            ema_loss = float(ema.update(loss.item()))
            pbar.set_postfix_str(f"loss={loss.item():.4f} ema={ema_loss:.4f} lr={lr:.2e}")

            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/loss_ema", ema_loss, global_step)
                writer.add_scalar("train/lr", lr, global_step)

        sched.step()

        acc = evaluate(model, test_dl, device)
        logger.info(f"Epoch {epoch:02d}: test acc {acc:.2f}% | lr {opt.param_groups[0]['lr']:.2e}")
        if writer is not None:
            writer.add_scalar("eval/acc", acc, epoch)

        # Save & push best
        if acc >= best_acc:
            best_acc = acc
            export = export_ternary_state(model)
            torch.save(export, args.export)
            logger.info(f"New best acc {best_acc:.2f}% — exported ternary state to {args.export}")

            # Write small metadata JSONs alongside the checkpoint
            meta_cfg = {
                "delta": cfg.delta,
                "clip_activations": cfg.clip_activations,
                "use_triton_linear": args.use_triton_linear,
            }
            metrics = {
                "epoch": epoch,
                "best_acc": best_acc,
                "lr": float(opt.param_groups[0]["lr"]),
                "global_step": global_step,
                "elapsed_sec": round(time.time() - start_time, 2),
            }
            cfg_json = "config.json"
            metrics_json = "metrics.json"
            dump_json(meta_cfg, cfg_json)
            dump_json(metrics, metrics_json)

            if hub_spec is not None:
                remote_pairs = [
                    (args.export, args.export),
                    (cfg_json, cfg_json),
                    (metrics_json, metrics_json),
                ]
                push_many(remote_pairs, hub_spec)
                logger.info("Pushed best checkpoint + metadata to 🤗 Hub.")

    logger.info(f"Training done. Best accuracy: {best_acc:.2f}%")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    train_main()
