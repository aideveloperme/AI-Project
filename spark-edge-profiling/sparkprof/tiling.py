"""Tiling experiment: a Triton GEMM with a sweepable tile shape.

For C[M,N] = A[M,K] @ B[K,N] computed in BLOCK_M x BLOCK_N output tiles, each tile
streams a BLOCK_M x K strip of A and a K x BLOCK_N strip of B. Ignoring L2 reuse, the
DRAM traffic is

    bytes ≈ elem * (M*N*K/BLOCK_N  +  M*N*K/BLOCK_M  +  M*N)
                   (A re-reads)        (B re-reads)      (C write)

so arithmetic intensity grows with the tile size: tiny tiles turn a compute-bound
GEMM into a memory-bound one. The sweep measures time, attained TFLOP/s and — with
Nsight Compute — real DRAM bytes, and compares them with this model (the gap is the
L2 cache doing its job). The same principle drives TensorRT's tactic / tiling choice
and the NCHW vs NHWC difference: layout decides which tiles are contiguous.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:
    @triton.jit
    def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                       stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                       GROUP_M: tl.constexpr):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        # grouped ordering (GROUP_M>1) improves L2 reuse; GROUP_M=1 is plain row-major
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c = acc.to(c_ptr.dtype.element_ty)
        cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + cm[:, None] * stride_cm + cn[None, :] * stride_cn
        tl.store(c_ptrs, c, mask=(cm[:, None] < M) & (cn[None, :] < N))


def tiled_matmul(a: torch.Tensor, b: torch.Tensor, bm: int, bn: int, bk: int,
                 group_m: int = 8, num_warps: int = 4, num_stages: int = 3) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                         c.stride(0), c.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                         GROUP_M=group_m, num_warps=num_warps, num_stages=num_stages)
    return c


def model_traffic_bytes(M: int, N: int, K: int, bm: int, bn: int, elem: int = 2) -> float:
    return elem * (M * N * K / bn + M * N * K / bm + M * N)


DEFAULT_TILES = [(16, 16, 32), (32, 32, 32), (64, 64, 32), (64, 128, 32), (128, 64, 32),
                 (128, 128, 32), (128, 128, 64), (128, 256, 64), (256, 128, 64)]
