"""
Triton kernels + autograd-backed ternary linear layer.

API
---
- pack_bitplanes(x): map int8 tensor in {-1,0,+1} to two uint32 planes (pos, neg) packed along K.
- ternary_gemm_triton(a_pos, a_neg, b_pos, b_neg, alpha_p, alpha_n, bias): GEMM producing float32 C.
- TernaryGemmFunction: autograd.Function using Triton in forward, dense STE surrogate in backward.
- TernaryLinearTriton: nn.Module drop-in linear layer with latent FP weights, αp/αn, bias; ternary
  input/output; forward uses popcount kernel, backward provides gradients for inputs, weights, scales.

Shapes
------
- GEMM uses row-major: A is [M, K], B is [N, K]; output C is [M, N] with C = A @ W_eff^T + b.
- Bit-packed planes are [dim0, ceil(K/32)] of dtype uint32.
"""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING, Protocol

import torch
import torch.nn as nn
import triton
import triton.language as tl

from btmnist.quant import TernaryConfig, ternary, TernaryActivation


def _ceil_div(a: int, b: int) -> int:
    """Integer ceil division."""
    return (a + b - 1) // b


def pack_bitplanes(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack a tensor of int8 values in {-1, 0, +1} into two uint32 bit-planes (pos, neg).

    Parameters
    ----------
    x : torch.Tensor
        2D tensor [R, K] int8 with entries in {-1, 0, +1}.

    Returns
    -------
    pos, neg : torch.Tensor
        Both uint32 tensors of shape [R, ceil(K/32)] representing +1 and -1 positions.
    """
    assert x.dtype == torch.int8 and x.dim() == 2
    device = x.device
    R, K = x.shape
    W = 32
    K_words = _ceil_div(K, W)
    pos = torch.zeros((R, K_words), dtype=torch.uint32, device=device)
    neg = torch.zeros((R, K_words), dtype=torch.uint32, device=device)

    # Vectorized within each 32-wide word
    arange32 = torch.arange(W, device=device, dtype=torch.uint32)
    for w in range(K_words):
        start = w * W
        end = min(start + W, K)
        width = end - start
        bits = (torch.ones((R, 1), device=device, dtype=torch.uint32) << arange32[:width]).contiguous()
        chunk = x[:, start:end]
        pos[:, w] = ((chunk == 1).to(torch.uint32) * bits).sum(dim=1)
        neg[:, w] = ((chunk == -1).to(torch.uint32) * bits).sum(dim=1)
    return pos, neg

@triton.jit
def _ternary_gemm_kernel(
    a_pos_ptr, a_neg_ptr,              # [M, Kw] uint32
    b_pos_ptr, b_neg_ptr,              # [N, Kw] uint32
    alpha_p_ptr, alpha_n_ptr,          # [N] float32
    bias_ptr,                          # [N] float32 (ignored if HAS_BIAS=False)
    c_ptr,                             # [M, N] float32
    M: int,
    N: int,
    Kw: int,
    lda: int,
    ldb: int,
    ldc: int,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KW: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc_pp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    acc_pn = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    acc_np = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    acc_nn = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for kw in range(0, Kw, BLOCK_KW):
        k_range = kw + tl.arange(0, BLOCK_KW)
        a_mask = (offs_m[:, None] < M) & (k_range[None, :] < Kw)
        b_mask = (offs_n[None, :] < N) & (k_range[:, None] < Kw)

        aP = tl.load(a_pos_ptr + offs_m[:, None] * lda + k_range[None, :], mask=a_mask, other=0)
        aN = tl.load(a_neg_ptr + offs_m[:, None] * lda + k_range[None, :], mask=a_mask, other=0)
        bP = tl.load(b_pos_ptr + offs_n[None, :] * ldb + k_range[:, None], mask=b_mask, other=0).T
        bN = tl.load(b_neg_ptr + offs_n[None, :] * ldb + k_range[:, None], mask=b_mask, other=0).T

        for t in range(0, BLOCK_KW):
            ap = aP[:, t]
            an = aN[:, t]
            bp = bP[t, :]
            bn = bN[t, :]
            acc_pp += tl.popcount(ap[:, None] & bp[None, :])
            acc_pn += tl.popcount(ap[:, None] & bn[None, :])
            acc_np += tl.popcount(an[:, None] & bp[None, :])
            acc_nn += tl.popcount(an[:, None] & bn[None, :])

    alpha_p = tl.load(alpha_p_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
    alpha_n = tl.load(alpha_n_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
    out = alpha_p * (acc_pp - acc_np).to(tl.float32) + alpha_n * (acc_nn - acc_pn).to(tl.float32)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
        out += bias

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * ldc + offs_n[None, :], out, mask=c_mask)

if TYPE_CHECKING: # 😎
    class _BTKernelLauncher(Protocol):
        def __call__(
            self,
            *,
            a_pos_ptr: torch.Tensor,
            a_neg_ptr: torch.Tensor,
            b_pos_ptr: torch.Tensor,
            b_neg_ptr: torch.Tensor,
            alpha_p_ptr: torch.Tensor,
            alpha_n_ptr: torch.Tensor,
            bias_ptr: torch.Tensor,
            c_ptr: torch.Tensor,
            M: int,
            N: int,
            Kw: int,
            lda: int,
            ldb: int,
            ldc: int,
            HAS_BIAS: bool,
            BLOCK_M: int,
            BLOCK_N: int,
            BLOCK_KW: int,
        ) -> None: ...

    class _BTKernel(Protocol):
        def __getitem__(self, grid: Tuple[int, int]) -> _BTKernelLauncher: ...

    _ternary_gemm_kernel_typed: _BTKernel
else:
    _ternary_gemm_kernel_typed = _ternary_gemm_kernel


def ternary_gemm_triton(
    a_pos: torch.Tensor,
    a_neg: torch.Tensor,
    b_pos: torch.Tensor,
    b_neg: torch.Tensor,
    alpha_p: torch.Tensor,
    alpha_n: torch.Tensor,
    bias: torch.Tensor | None = None,
    block: int = 128,
    block_kw: int = 8,
) -> torch.Tensor:
    assert a_pos.dtype == torch.uint32 and a_neg.dtype == torch.uint32
    assert b_pos.dtype == torch.uint32 and b_neg.dtype == torch.uint32
    assert a_pos.device == b_pos.device == alpha_p.device
    M, Kw = a_pos.shape
    N, Kw_b = b_pos.shape
    assert Kw == Kw_b, "Mismatched K words between A and B."

    C = torch.empty((M, N), device=a_pos.device, dtype=torch.float32)
    grid = (triton.cdiv(M, block), triton.cdiv(N, block))

    _ternary_gemm_kernel_typed[grid](
        a_pos_ptr=a_pos,
        a_neg_ptr=a_neg,
        b_pos_ptr=b_pos,
        b_neg_ptr=b_neg,
        alpha_p_ptr=alpha_p,
        alpha_n_ptr=alpha_n,
        bias_ptr=bias if bias is not None else alpha_p,
        c_ptr=C,
        M=M,
        N=N,
        Kw=Kw,
        lda=Kw,
        ldb=Kw,
        ldc=N,
        HAS_BIAS=(bias is not None),
        BLOCK_M=block,
        BLOCK_N=block,
        BLOCK_KW=block_kw,
    )

    return C



class TernaryGemmFunction(torch.autograd.Function):
    """
    Autograd wrapper for ternary GEMM.

    Forward:
        - Ternarize A (activations) and latent W (weights) using delta.
        - Bit-pack A and W into two planes each.
        - Triton kernel computes C = A @ W_eff^T + b with W_eff = αp*P - αn*N.

    Backward (STE surrogate):
        - dA_pre = dC @ W_eff, gated by activation STE mask 1{|A_pre|≤1}.
        - dW_eff = dC^T @ A_tern.
        - Map to parameters:
            * dαp[n] = sum_k dW_eff[n,k] * P[n,k]
            * dαn[n] = -sum_k dW_eff[n,k] * N[n,k]
            * dW_latent[n,k] = 1{|W_latent|≤1} * dW_eff[n,k] * (αp if q==+1, αn if q==-1, else 0)
        - dbias = row-wise sum over dC.
    """

    @staticmethod
    def forward(ctx,
                A_pre: torch.Tensor,        # [B, K] float (pre-activation)
                W_latent: torch.Tensor,     # [N, K] float
                alpha_p: torch.Tensor,      # [N] float
                alpha_n: torch.Tensor,      # [N] float
                bias: torch.Tensor | None,  # [N] float or None
                delta: float) -> torch.Tensor:

        device = A_pre.device
        # Ternarize inputs
        A_tern = ternary(torch.clamp(A_pre, -2.5, 2.5), delta).to(torch.int8)  # [B, K]
        W_q = ternary(W_latent, delta).to(torch.int8)                           # [N, K]

        # Pack bit-planes
        a_pos, a_neg = pack_bitplanes(A_tern)
        b_pos, b_neg = pack_bitplanes(W_q)

        # Triton forward
        C = ternary_gemm_triton(a_pos, a_neg, b_pos, b_neg, alpha_p, alpha_n, bias)

        # Save tensors needed for backward (store small/int where possible)
        ctx.save_for_backward(
            A_pre,                       # float
            A_tern.to(torch.float32),    # keep as float {-1,0,1} for matmuls
            W_latent,                    # float
            W_q,                         # int8 ternary weights
            alpha_p, alpha_n,
            bias if bias is not None else torch.tensor([], device=device)
        )
        ctx.delta = delta
        return C

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Unpack single upstream gradient
        (grad_out,) = grad_outputs

        A_pre, A_tern_f, W_latent, W_q_i8, alpha_p, alpha_n, bias = ctx.saved_tensors
        _delta = ctx.delta

        # Masks and effective weights
        P = (W_q_i8 == 1).to(grad_out.dtype)       # [N, K]
        Nmask = (W_q_i8 == -1).to(grad_out.dtype)  # [N, K]
        W_eff = alpha_p.view(-1, 1) * P - alpha_n.view(-1, 1) * Nmask  # [N, K]

        # dA_pre (apply activation STE mask)
        dA = grad_out @ W_eff                             # [B, K]
        a_mask = (A_pre.abs() <= 1.0).to(dA.dtype)
        dA_pre = dA * a_mask

        # dW_eff
        dW_eff = grad_out.t() @ A_tern_f                  # [N, K]

        # d alpha_p / alpha_n
        d_alpha_p = (dW_eff * P).sum(dim=1)              # [N]
        d_alpha_n = -(dW_eff * Nmask).sum(dim=1)         # [N]

        # d W_latent via STE (gate by |W_latent|≤1, scale by branch α)
        w_mask = (W_latent.abs() <= 1.0).to(dW_eff.dtype)
        branch_scale = torch.where(
            P.bool(), alpha_p.view(-1, 1),
            torch.where(Nmask.bool(), alpha_n.view(-1, 1), torch.zeros_like(W_eff))
        )
        dW_latent = w_mask * dW_eff * branch_scale

        # dbias
        d_bias = grad_out.sum(dim=0) if bias.numel() > 0 else None

        # Return grads for each forward arg (A_pre, W_latent, alpha_p, alpha_n, bias, delta)
        return dA_pre, dW_latent, d_alpha_p, d_alpha_n, d_bias, None



class TernaryLinearTriton(nn.Module):
    """
    Linear layer with latent FP weights and TTQ scales; forward uses Triton popcount kernel;
    backward uses STE surrogate (see TernaryGemmFunction).

    Parameters
    ----------
    in_f : int
        Input features.
    out_f : int
        Output features.
    cfg : TernaryConfig
        Ternarization config (delta, clip).
    bias : bool
        Whether to use bias.
    act : bool
        Whether to apply a ternary activation to the output (to match the pure-ternary motif).
    """

    def __init__(self, in_f: int, out_f: int, cfg: TernaryConfig, bias: bool = True, act: bool = True):
        super().__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.delta = cfg.delta
        self.preclip = cfg.clip_activations
        self.use_bias = bias
        self.act = TernaryActivation(cfg) if act else nn.Identity()

        self.w_latent = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_uniform_(self.w_latent, a=2.0 ** 0.5)
        self.alpha_p = nn.Parameter(torch.ones(out_f))
        self.alpha_n = nn.Parameter(torch.ones(out_f))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = TernaryGemmFunction.apply(
            x, self.w_latent, self.alpha_p, self.alpha_n, self.bias, self.delta
        )
        return self.act(y)
