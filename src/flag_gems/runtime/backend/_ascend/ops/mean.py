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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

@libentry()
@triton.jit
def mean_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < M
    inp_val = tl.load(inp + off, mask=mask, other=0.0).to(tl.float32)
    partial_sum = tl.sum(inp_val, axis=0)
    tl.store(mid + pid, partial_sum)


def mean(inp, *, dtype=None):
    logger.debug("GEMS_ASCEND MEAN")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    block_size = min(block_size, 2048)
    out = torch.zeros([], dtype=torch.float32, device=inp.device)
    num_ctas = triton.cdiv(M, block_size)
    mid = torch.zeros([num_ctas], dtype=torch.float32, device=inp.device)

    with torch_device_fn.device(inp.device):
        mean_kernel_1[(num_ctas, 1, 1)](inp, mid, M, block_size)
        out = mid.sum() / M
    return out.to(dtype)

def choose_mean_dim_tiling(M, N):
    MAX_COREDIM = 65535
    TARGET_PROGRAMS = 8192
    MAX_BLOCK_N = 256
    MAX_SUB = 256
    ACC_BUDGET_BYTES = 64 * 1024

    block_n = min(triton.next_power_of_2(N), MAX_BLOCK_N)

    max_sub_by_budget = ACC_BUDGET_BYTES // (block_n * 4)
    block_m_sub = min(
        MAX_SUB,
        1 << (max_sub_by_budget.bit_length() - 1),
    )

    block_m_main = triton.next_power_of_2(
        max(
            block_m_sub,
            triton.cdiv(M, TARGET_PROGRAMS),
            triton.cdiv(M, MAX_COREDIM),
        )
    )

    if block_m_main > block_m_sub:
        block_m_sub = min(block_m_sub, 128)

    return block_m_main, block_m_sub, block_n

@libentry()
@triton.jit
def mean_dim_kernel(
    X,
    Mean,
    M,
    N,
    BLOCK_M_MAIN: tl.constexpr,
    BLOCK_M_SUB: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    base_row = ext.program_id(0) * BLOCK_M_MAIN

    for row_off in range(0, BLOCK_M_MAIN, BLOCK_M_SUB):
        rows = base_row + row_off + tl.arange(0, BLOCK_M_SUB)[:, None]
        row_mask = rows < M

        row_x = X + rows * N
        acc = tl.zeros(
            [BLOCK_M_SUB, BLOCK_N],
            dtype=tl.float32,
        )

        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            mask = row_mask & (cols < N)

            vals = tl.load(
                row_x + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

            acc += vals

        means = tl.sum(acc, axis=1) / N

        tl.store(
            Mean + rows,
            means[:, None],
            mask=row_mask,
        )


def mean_dim(x, dim, keepdim=False, *, dtype=None):
    logger.debug("GEMS_ASCEND MEAN_DIM")

    if dtype is None:
        dtype = x.dtype
    if dim is None:
        out = mean(x, dtype=dtype)
        if not keepdim:
            out = out.reshape([1] * x.ndim)
        return out

    shape = list(x.shape)
    dim = [d % x.ndim for d in dim]
    x = dim_compress(x, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = x.numel() // N
    out = torch.empty(shape, dtype=dtype, device=x.device)
    block_m_main, block_m_sub, block_n = choose_mean_dim_tiling(M, N)

    grid = (triton.cdiv(M, block_m_main),)
    
    with torch_device_fn.device(x.device):
        mean_dim_kernel[grid](
            x,
            out,
            M,
            N,
            BLOCK_M_MAIN=block_m_main,
            BLOCK_M_SUB=block_m_sub,
            BLOCK_N=block_n,
            num_warps=4,
            num_stages=2,
        ) 
    if not keepdim:
        out = out.squeeze(dim)
    return out
