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
import os
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from .sum import sum_dim

logger = logging.getLogger(__name__)

# Ascend NPU coreDim (the launch grid) cannot exceed UINT16_MAX. Ops whose grid
# is derived from the row count must cap it and walk the remaining work inside
# the kernel, see triton-ascend docs migration_guide/migrate_from_gpu.md.
MAX_CORE_DIM = 65535

# 0：自动选择；其他值：强制指定，供独立测试使用。
_RMS_FWD_BLOCK_M = int(os.getenv("FG_RMS_FWD_BLOCK_M", "0"))
_RMS_FWD_BLOCK_M_SUB = int(os.getenv("FG_RMS_FWD_BLOCK_M_SUB", "0"))

# A conservative live-array budget, NOT the hardware UB capacity. Reserve room
# for compiler temporaries/multibuffering; warmup below checks the actual UB use.
_RMS_FWD_ARRAY_BUDGET_BYTES = 144 * 1024
_RMS_FWD_TILE_CACHE = {}

# Ascend limits the product of all launch-grid dimensions.
_ASCEND_MAX_CORE_DIM = 65535


def _floor_power_of_two(value):
    return 1 << (max(1, value).bit_length() - 1)


@lru_cache(maxsize=None)
def _rms_norm_vector_cores(device_index):
    cores = int(torch_device_fn.get_device_properties(device_index).vector_core_num)
    if cores < 1:
        raise ValueError("RMSNorm requires a positive vector core count")
    return cores


def _select_rms_norm_fwd_tiles(M, N, x_bytes, w_bytes, vector_cores,
                               block_m=0, block_m_sub=0):
    """Compute tiles without enumerating shapes; explicit test overrides win."""
    for name, value in (("BLOCK_M", block_m), ("BLOCK_M_SUB", block_m_sub)):
        if value < 0 or (value and value & (value - 1)):
            raise ValueError(f"FG_RMS_FWD_{name} must be 0 or a positive power of two")

    block_size = min(triton.next_power_of_2(N), 4096)
    # Approximate three FP32 arrays plus input/output storage per tile element.
    # Weight storage is per column, not per row. This is an estimate, not a
    # compiler liveness model; only compilation can establish actual UB use.
    row_bytes = block_size * (3 * 4 + 2 * x_bytes)
    available = max(row_bytes, _RMS_FWD_ARRAY_BUDGET_BYTES - block_size * w_bytes)
    resident_rows = _floor_power_of_two(available // row_bytes)

    # Keep at least one wave of row tiles when M permits. A logical tile may
    # process two resident subtiles sequentially, without doubling live arrays.
    parallel_rows = _floor_power_of_two(max(1, M // vector_cores))
    if block_m == 0:
        block_m = min(2 * resident_rows, parallel_rows)
        if block_m_sub:
            block_m = max(block_m, block_m_sub)
    if block_m_sub == 0:
        block_m_sub = min(resident_rows, block_m)
    if block_m_sub > block_m:
        raise ValueError("FG_RMS_FWD_BLOCK_M_SUB must not exceed BLOCK_M")
    return block_size, block_m, block_m_sub




@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_kernel(
    Y,
    INV_RMS,
    X,
    W,
    y_stride_r,
    y_stride_c,
    x_stride_r,
    x_stride_c,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr = 1,
    BLOCK_M_SUB: tl.constexpr = 0,
):
    # Logical rows per program and resident rows are independently configurable.
    ROWS_PER_SUB: tl.constexpr = BLOCK_M if BLOCK_M_SUB == 0 else BLOCK_M_SUB
    row_offsets = tl.arange(0, ROWS_PER_SUB)
    num_row_tiles = tl.cdiv(M, BLOCK_M)
    for tile in range(ext.program_id(0), num_row_tiles, ext.num_programs(0)):
        for row_sub in range(0, BLOCK_M, ROWS_PER_SUB):
            rows = tile * BLOCK_M + row_sub + row_offsets
            row_mask = (rows < M) & (row_sub + row_offsets < BLOCK_M)
            # Keep only the current sub-block's row statistics live.
            var = tl.full((ROWS_PER_SUB,), 0.0, tl.float32)
            for off in range(0, N, BLOCK_SIZE):
                cols = off + tl.arange(0, BLOCK_SIZE)
                mask = row_mask[:, None] & (cols[None, :] < N)
                x = tl.load(
                    X + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
                    mask,
                    other=0.0,
                ).to(tl.float32)
                var += tl.sum(x * x / N, axis=1)

            rrms = 1 / tl.sqrt(var + eps)

            for off in range(0, N, BLOCK_SIZE):
                cols = off + tl.arange(0, BLOCK_SIZE)
                mask = row_mask[:, None] & (cols[None, :] < N)
                x = tl.load(
                    X + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
                    mask,
                    other=0.0,
                ).to(tl.float32)
                w = tl.load(W + cols, cols < N, other=0.0)
                y_fp32 = x * rrms[:, None] * w[None, :].to(tl.float32)
                y = y_fp32.to(Y.dtype.element_ty)
                tl.store(
                    Y + rows[:, None] * y_stride_r + cols[None, :] * y_stride_c,
                    y,
                    mask=mask,
                )

            tl.store(INV_RMS + rows, rrms, mask=row_mask)



class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        y, inv_rms = rms_norm_forward(x, normalized_shape, weight, eps)

        ctx.save_for_backward(x, inv_rms, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, inv_rms, weight = ctx.saved_tensors
        dx, dw = rms_norm_backward(
            dy, x, inv_rms, ctx.normalized_shape, weight, ctx.eps
        )
        return dx, None, dw, None


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return RmsNorm.apply(x, normalized_shape, weight, eps)


def rms_norm_forward(x, normalized_shape, weight, eps=1e-5):
    logger.debug("GEMS_ASCEND RMS_NORM_FORWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    x = x.contiguous()
    weight = weight.contiguous()
    y = torch.empty_like(x)
    inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)

    if M:
        device_index = x.device.index
        vector_cores = _rms_norm_vector_cores(device_index)
        # This cache is populated at runtime after successful compilation;
        # it contains no hand-written shape/configuration pairs.
        key = (device_index, M, N, x.dtype, weight.dtype, type(eps),
               x.data_ptr() % 16, weight.data_ptr() % 16,
               _RMS_FWD_BLOCK_M, _RMS_FWD_BLOCK_M_SUB)
        tiles = _RMS_FWD_TILE_CACHE.get(key)
        with torch_device_fn.device(x.device):
            if tiles is None:
                block_size, block_m, block_m_sub = _select_rms_norm_fwd_tiles(
                    M, N, x.element_size(), weight.element_size(), vector_cores,
                    _RMS_FWD_BLOCK_M, _RMS_FWD_BLOCK_M_SUB,
                )
                automatic = _RMS_FWD_BLOCK_M == 0 and _RMS_FWD_BLOCK_M_SUB == 0
                while True:
                    grid = (min(triton.cdiv(M, block_m), _ASCEND_MAX_CORE_DIM),)
                    try:
                        # Compile only: do not launch a potentially oversized tile.
                        rms_norm_kernel.fn.warmup(
                            y, inv_rms, x, weight, N, 1, N, 1, M, N, eps,
                            block_size, block_m, block_m_sub, grid=grid,
                        )
                        break
                    except Exception as exc:
                        # Other compiler errors and all execution errors propagate.
                        if not automatic or "ub overflow" not in str(exc).lower():
                            raise
                        previous = (block_size, block_m, block_m_sub)
                        if block_m_sub > 1:
                            block_m_sub //= 2
                        elif block_size > 1:
                            block_size //= 2
                        else:
                            raise
                        logger.warning("RMSNorm UB compile backoff: %s -> %s",
                                       previous, (block_size, block_m, block_m_sub))
                tiles = (block_size, block_m, block_m_sub)
                _RMS_FWD_TILE_CACHE[key] = tiles
            block_size, block_m, block_m_sub = tiles
            grid = (min(triton.cdiv(M, block_m), _ASCEND_MAX_CORE_DIM),)
            rms_norm_kernel[grid](
                y, inv_rms, x, weight, N, 1, N, 1, M, N, eps,
                block_size, block_m, block_m_sub,
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
        dw = sum_dim(partial_buffer, dim=[0], dtype=x.dtype)

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
