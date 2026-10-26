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


@libentry()
@triton.jit(do_not_specialize=["eps", "M"])
def rms_norm_grad_dx_kernel(
    X,  # pointer to the input
    DY,
    INV_RMS,  # pointer to inverse rms
    DX,  # pointer to the output
    W,  # pointer to the weights
    dx_stride_r,
    dx_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # The grid is capped at MAX_CORE_DIM, so each program handles several rows.
    for row in range(ext.program_id(0), M, ext.num_programs(0)):
        dx_row = DX + row * dx_stride_r
        x_row = X + row * x_stride_r
        dy_row = DY + row * x_stride_r

        inv_rms = tl.load(INV_RMS + row).to(tl.float32)

        row_sum_stats = 0.0
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(x_row + cols, mask, other=0.0).to(tl.float32)
            dy = tl.load(dy_row + cols, mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask, other=0.0).to(tl.float32)
            dy = dy * w
            normalized_buf = x * inv_rms
            row_sum_stats += tl.sum(normalized_buf * dy)

        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(x_row + cols, mask, other=0.0).to(tl.float32)
            dy = tl.load(dy_row + cols, mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask, other=0.0).to(tl.float32)
            dy = dy * w
            normalized_buf = x * inv_rms
            norm_val = normalized_buf / N
            dx = (dy - norm_val * row_sum_stats) * inv_rms
            tl.store(dx_row + cols * dx_stride_c, dx, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "num_col_blocks"])
def rms_norm_grad_dw_kernel(
    X,  # pointer to the input
    DY,
    INV_RMS,  # pointer to inverse rms
    DW,  # pointer to the output
    dx_stride_r,
    dx_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    M,  # number of rows in X
    N,  # number of columns in X
    num_col_blocks,  # number of col tiles per row tile
    ROW_BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
):
    rows = tl.arange(0, ROW_BLOCK_SIZE)
    cols = tl.arange(0, COL_BLOCK_SIZE)

    # The (row tile, col tile) grid is flattened to 1D so that it can be capped
    # at MAX_CORE_DIM; each program then walks several tiles.
    num_tiles = tl.cdiv(M, ROW_BLOCK_SIZE) * num_col_blocks
    for tile in range(ext.program_id(0), num_tiles, ext.num_programs(0)):
        row_pid = tile // num_col_blocks
        col_pid = tile % num_col_blocks

        row_start = row_pid * ROW_BLOCK_SIZE
        col_start = col_pid * COL_BLOCK_SIZE

        # int64: the element offset of the last row tile easily exceeds int32.
        offset = row_start.to(tl.int64) * x_stride_r + col_start * x_stride_c

        row_mask = (row_start + rows) < M
        col_mask = (col_start + cols) < N

        x = tl.load(
            X + offset + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
            row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        inv_rms = tl.load(INV_RMS + row_start + rows, row_mask, other=0.0).to(
            tl.float32
        )
        dy = tl.load(
            DY + offset + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
            row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        d_weight = x * dy * inv_rms[:, None]
        partial_dweight_sum = tl.sum(d_weight, axis=0)

        tl.store(
            DW + row_pid * N + col_start + cols,
            partial_dweight_sum,
            mask=col_mask,
        )


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
    """Backward of rms_norm_forward, returning (dx, dw)."""
    logger.debug("GEMS_ASCEND RMS_NORM_BACKWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    BLOCK_SIZE = min(triton.next_power_of_2(N), 6912)
    x = x.contiguous()
    dy = dy.contiguous()
    weight = weight.contiguous()
    dx = torch.empty_like(x)

    with torch_device_fn.device(x.device):
        rms_norm_grad_dx_kernel[min(M, MAX_CORE_DIM),](
            x, dy, inv_rms, dx, weight, N, 1, N, 1, M, N, eps, BLOCK_SIZE
        )

    ROW_BLOCK_SIZE = 16
    COL_BLOCK_SIZE = 256
    row_block_num = triton.cdiv(M, ROW_BLOCK_SIZE)
    col_block_num = triton.cdiv(N, COL_BLOCK_SIZE)

    partial_buffer = torch.empty(
        (row_block_num, N), dtype=torch.float32, device=x.device
    )

    num_tiles = row_block_num * col_block_num
    with torch_device_fn.device(x.device):
        rms_norm_grad_dw_kernel[min(num_tiles, MAX_CORE_DIM),](
            x,
            dy,
            inv_rms,
            partial_buffer,
            N,
            1,
            N,
            1,
            M,
            N,
            col_block_num,
            ROW_BLOCK_SIZE,
            COL_BLOCK_SIZE,
        )
        dw = sum_dim(partial_buffer, dim=[0], dtype=x.dtype)

    return dx, dw
