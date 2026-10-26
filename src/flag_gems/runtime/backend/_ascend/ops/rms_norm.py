# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Ascend NPU coreDim (the launch grid) cannot exceed UINT16_MAX. Ops whose grid
# is derived from the row count must cap it and walk the remaining work inside
# the kernel, see triton-ascend docs migration_guide/migrate_from_gpu.md.
MAX_CORE_DIM = 65535


@libentry()
@triton.jit(do_not_specialize=["eps", "M"])
def rms_norm_kernel(
    Y,  # pointer to the output
    INV_RMS,  # pointer to inverse rms
    X,  # pointer to the input
    W,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # The grid is capped at MAX_CORE_DIM, so each program handles several rows.
    for row in range(ext.program_id(0), M, ext.num_programs(0)):
        x_row = X + row * x_stride_r
        y_row = Y + row * y_stride_r

        var = 0.0
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(x_row + cols, mask, other=0.0).to(tl.float32)
            var += tl.sum(x * x / N)

        rrms = 1 / tl.sqrt(var + eps)

        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(x_row + cols, mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask, other=0.0)
            y = (x * rrms).to(Y.dtype.element_ty) * w
            tl.store(y_row + cols * y_stride_c, y, mask=mask)

        tl.store(INV_RMS + row, rrms)



class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        logger.debug("GEMS_ASCEND LAYERNORM_FORWARD")
        y, inv_rms = rms_norm_forward(x, normalized_shape, weight, eps)

        ctx.save_for_backward(x, inv_rms, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        logger.debug("GEMS_ASCEND LAYERNORM_BACKWARD")
        x, inv_rms, weight = ctx.saved_tensors
        dx, dw = rms_norm_backward(
            dy, x, inv_rms, ctx.normalized_shape, weight, ctx.eps
        )
        return dx, None, dw, None


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return RmsNorm.apply(x, normalized_shape, weight, eps)


def rms_norm_forward(x, normalized_shape, weight, eps=1e-5):
    """rms_norm without autograd, returning (y, inv_rms).

    TransformerEngine's FlagOS backend calls this entry point directly, so it
    must carry the same capped launch as RmsNorm.forward or that path still
    overflows coreDim.
    """
    logger.debug("GEMS_ASCEND RMS_NORM_FORWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    BLOCK_SIZE = min(triton.next_power_of_2(N), 12064)

    x = x.contiguous()
    weight = weight.contiguous()
    y = torch.empty_like(x)
    inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)

    with torch_device_fn.device(x.device):
        rms_norm_kernel[min(M, MAX_CORE_DIM),](
            y, inv_rms, x, weight, N, 1, N, 1, M, N, eps, BLOCK_SIZE
        )

    return y, inv_rms


def rms_norm_backward(dy, x, inv_rms, normalized_shape, weight, eps=1e-5):
    """Narrow-row backward: fused dx/partial dw, followed by dw reduction."""
    logger.debug("GEMS_ASCEND RMS_NORM_BACKWARD")
    if not 0 < len(normalized_shape) <= x.ndim:
        raise ValueError("Invalid normalized_shape")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)
    if x.shape != dy.shape:
        raise ValueError("x and dy must have the same shape")
    if tuple(x.shape[dim:]) != tuple(normalized_shape):
        raise ValueError("normalized_shape does not match x")
    if tuple(weight.shape) != tuple(normalized_shape):
        raise ValueError("weight shape does not match normalized_shape")
    if inv_rms.numel() != M:
        raise ValueError(f"Expected inv_rms.numel() == M ({M})")
    if M == 0 or N == 0:
        raise ValueError("Empty tensors are not supported yet")

    x = x.contiguous()
    dy = dy.contiguous()
    weight = weight.contiguous()
    inv_rms = inv_rms.contiguous()
    dx = torch.empty_like(x)
    rows_per_program = (M + 47) // 48
    grid = ((M + rows_per_program - 1) // rows_per_program,)
    dw = torch.zeros((N,), device=x.device, dtype=torch.float32)

    if N > 6080:
        rms_norm_grad_kernel = rms_norm_grad_split_d_high_precision
        tail_n = N % 4096
        block_tail = tail_n if tail_n else 1
        extra_args = (4096, block_tail)

        multibuffer = False
    
    elif (
        N <= 640
        and x.dtype == torch.float32
        and dy.dtype == torch.float32
        and weight.dtype == torch.float32
        and inv_rms.dtype == torch.float32
    ):
        block_n = 1 << (N - 1).bit_length()
        tile_rows = min(rows_per_program, 16, 2048 // block_n)
        block_m = 1 << (tile_rows.bit_length() - 1)
    
        rms_norm_grad_kernel = rms_norm_grad_split_n_smalld
        extra_args = (block_m, block_n)
        multibuffer = False
    
    else:
        rms_norm_grad_kernel = rms_norm_grad_split_n_high_precision

        # FlagTree Ascend supports exact-width vectors; avoid power-of-two padding.
        extra_args = (N,)

        multibuffer = (N <= 4096)

    with torch_device_fn.device(x.device):
        rms_norm_grad_kernel[grid](
            dy, x, inv_rms, weight, dx, dw,
            M, N, rows_per_program, *extra_args,
            multibuffer=multibuffer,
        )
    return dx, dw


@libentry()
@triton.jit
def rms_norm_grad_split_n_high_precision(
    DY, X, InvRms, Weight, DX, DW,
    M, N: tl.constexpr, ROWS_PER_PROGRAM: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Each program owns rows and accumulates its FP32 partial weight gradient."""
    pid = tl.program_id(0)
    row_begin = pid * ROWS_PER_PROGRAM
    row_end = tl.minimum(row_begin + ROWS_PER_PROGRAM, M)

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    gamma = tl.load(Weight + cols, mask=mask, other=0).to(tl.float32)
    dw_acc = tl.full((BLOCK_N,), 0, tl.float32)
    for row in range(row_begin, row_end):
        x = tl.load(X + row * N + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + row * N + cols, mask=mask, other=0).to(tl.float32)
        rstd = tl.load(InvRms + row).to(tl.float32)
        z = x * rstd
        g = dy * gamma
        dw_acc += dy * z
        mean = tl.sum(g * z, 0) / N
        dx = rstd * (g - z * mean)
        tl.store(DX + row * N + cols, dx, mask=mask)
    tl.atomic_add(DW + cols, dw_acc, mask=mask)
 


@libentry()
@triton.jit
def rms_norm_grad_split_n_smalld(
    DY, X, InvRms, Weight, DX, DW,
    M, N: tl.constexpr, ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    tile_rows = tl.arange(0, BLOCK_M)
    valid_cols = cols < N
    gamma = tl.load(Weight + cols, mask=valid_cols, other=0).to(tl.float32)
    dw_acc = tl.full((BLOCK_N,), 0, tl.float32)
    row_begin = pid * ROWS_PER_PROGRAM

    # 逐 Tile 遍历当前 Program 负责的行，末尾行使用 mask 防止越界。
    for offset in range(0, ROWS_PER_PROGRAM, BLOCK_M):
        rows = row_begin + offset + tile_rows
        valid_rows = (rows < M) & (rows < row_begin + ROWS_PER_PROGRAM)
        mask = valid_rows[:, None] & valid_cols[None, :]
        indices = rows[:, None] * N + cols[None, :]
        x = tl.load(X + indices, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + indices, mask=mask, other=0).to(tl.float32)
        rstd = tl.load(InvRms + rows, mask=valid_rows, other=0).to(tl.float32)
        z = x * rstd[:, None]
        g = dy * gamma[None, :]

        # 与 CANN ComputeSmallD 一致：dw 沿行归约，dx 的均值逐行沿列归约。
        dw_acc += tl.sum(dy * z, axis=0)
        mean = tl.sum(g * z, axis=1) / N
        dx = rstd[:, None] * (g - z * mean[:, None])
        tl.store(DX + indices, dx, mask=mask)

    tl.atomic_add(DW + cols, dw_acc, mask=valid_cols, sem="relaxed")

@triton.jit
def _splitd_before_tile(
    DY, X, InvRms, Weight, DW, N: tl.constexpr,
    group_begin, group_rows, row_slots, row_dots,
    col_start, COL_TILE: tl.constexpr,
):
    cols = col_start + tl.arange(0, COL_TILE)
    valid_cols = cols < N
    gamma = tl.load(Weight + cols, mask=valid_cols, other=0).to(tl.float32)
    dw_acc = tl.full((COL_TILE,), 0, tl.float32)
    for local_row in range(0, group_rows):
        row = group_begin + local_row
        base = row * N + cols
        x = tl.load(X + base, mask=valid_cols, other=0).to(tl.float32)
        dy = tl.load(DY + base, mask=valid_cols, other=0).to(tl.float32)
        rstd = tl.load(InvRms + row).to(tl.float32)
        dy_z = dy * (x * rstd)
        dw_acc += dy_z
        partial_dot = tl.sum(dy_z * gamma, axis=0)
        row_dots += tl.where(row_slots == local_row, partial_dot, 0.0)
    tl.atomic_add(DW + cols, dw_acc, mask=valid_cols, sem="relaxed")
    return row_dots


@triton.jit
def _splitd_after_tile(
    DY, X, InvRms, Weight, DX, N: tl.constexpr,
    group_begin, group_rows, row_slots, row_means,
    col_start, COL_TILE: tl.constexpr,
):
    cols = col_start + tl.arange(0, COL_TILE)
    valid_cols = cols < N
    gamma = tl.load(Weight + cols, mask=valid_cols, other=0).to(tl.float32)
    for local_row in range(0, group_rows):
        row = group_begin + local_row
        mean = tl.reshape(tl.extract_slice(row_means, (local_row,), (1,), (1,)), ())
        base = row * N + cols
        x = tl.load(X + base, mask=valid_cols, other=0).to(tl.float32)
        dy = tl.load(DY + base, mask=valid_cols, other=0).to(tl.float32)
        rstd = tl.load(InvRms + row).to(tl.float32)
        z = x * rstd
        g = dy * gamma
        dx = rstd * (g - z * mean)
        tl.store(DX + base, dx, mask=valid_cols)


@libentry()
@triton.jit
def rms_norm_grad_split_d_high_precision(
    DY, X, InvRms, Weight, DX, DW,
    M, N: tl.constexpr, ROWS_PER_PROGRAM: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_TAIL: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    row_begin = pid * ROWS_PER_PROGRAM
    row_end = tl.minimum(row_begin + ROWS_PER_PROGRAM, M)
    ROW_TILE: tl.constexpr = 16
    row_slots = tl.arange(0, ROW_TILE)
    FULL_COLS: tl.constexpr = (N // BLOCK_D) * BLOCK_D

    for group_begin in range(row_begin, row_end, ROW_TILE):
        group_rows = tl.minimum(ROW_TILE, row_end - group_begin)
        row_dots = tl.full((ROW_TILE,), 0, tl.float32)

        for col_start in range(0, FULL_COLS, BLOCK_D):
            row_dots = _splitd_before_tile(
                DY, X, InvRms, Weight, DW, N,
                group_begin, group_rows, row_slots, row_dots,
                col_start, BLOCK_D,
            )
        if N > FULL_COLS:
            row_dots = _splitd_before_tile(
                DY, X, InvRms, Weight, DW, N,
                group_begin, group_rows, row_slots, row_dots,
                FULL_COLS, BLOCK_TAIL,
            )

        row_means = row_dots / N

        for col_start in range(0, FULL_COLS, BLOCK_D):
            _splitd_after_tile(
                DY, X, InvRms, Weight, DX, N,
                group_begin, group_rows, row_slots, row_means,
                col_start, BLOCK_D,
            )
        if N > FULL_COLS:
            _splitd_after_tile(
                DY, X, InvRms, Weight, DX, N,
                group_begin, group_rows, row_slots, row_means,
                FULL_COLS, BLOCK_TAIL,
            )
