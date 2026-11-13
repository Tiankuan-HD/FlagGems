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

import contextlib
import logging
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend import heuristics_config_utils as _hcu
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _blockified():
    previous = os.environ.get("TRITON_ALL_BLOCKS_PARALLEL")
    os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TRITON_ALL_BLOCKS_PARALLEL", None)
        else:
            os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = previous


@contextlib.contextmanager
def _compiling(device):
    """Scope a launch that may compile: bind the device, ask for blockified
    code. A cached launch does neither, so it skips this."""
    with torch_device_fn.device(device), _blockified():
        yield


# A `mm_kernel` launch costs 110-130 us of host time here against 10-14 us for
# the same GEMM through aten, while the kernels are within 1.2x. Taking one
# launch apart shows the hardware launch is 29 us of that; the rest is Python
# that `LibEntry.run` rebuilds every call -- it reclassifies each argument,
# rebuilds the dispatch key, and re-evaluates the grid.
#
# For this kernel all three results are a pure function of the key below:
# the grid reads only M, N and the selected tile, and M, N and every stride are
# explicit specialized arguments. So the assembled launch is stored once per
# key and a hit goes straight to the compiled kernel. `_nvidia/hopper/ops/
# mm_w8a8_fp8.py` keeps a launch cache the same way.
#
# A miss still goes through `LibEntry.run` unchanged, so compilation,
# autotuning and heuristics behave exactly as before; only a hit takes the
# short path, and it neither compiles nor rebinds the device. A stored entry is
# not revalidated against the tuner -- the same lifetime `LibEntry` already
# gives the compiled kernels in its own cache.
_MM_LAUNCH_CACHE = {}
_MM_LAUNCH_LIMIT = 1024
# Bound to module level so one `site` implies one fixed set of keyword
# arguments and the key does not have to carry them.
# GROUP_M is deliberately absent from both: triton rejects a keyword argument
# that is also an auto-tuned config key, and on weight-gradient shapes the
# tuner's choice is worth ~30% (see `_prune_mm_configs`).
_MM_KWARGS = {
    "dot_out_dtype": tl.float32,
    "multibuffer": True,
    "unit_flag": True,
    "sync_solver": False,
}
_MM_OUT_KWARGS = {"dot_out_dtype": tl.float32}


def _launch_mm_kernel(args, kwargs, site, grid):
    """Launch `mm_kernel`, reusing a cached assembly when the key matches.

    `args[:3]` are the operands, substituted on every call; everything after
    them -- sizes and strides -- enters the key by value. `site` separates call
    sites, each of which passes one fixed `kwargs`.
    """
    a, b, c = args[0], args[1], args[2]
    key = (
        site,
        a.dtype,
        b.dtype,
        c.dtype,
        a.device.index,
        a.data_ptr() % 16,
        b.data_ptr() % 16,
        c.data_ptr() % 16,
    ) + tuple(args[3:])

    plan = _MM_LAUNCH_CACHE.get(key)
    if plan is None:
        with _compiling(a.device):
            kernel, constexprs = mm_kernel[grid](*args, **kwargs)
        # Reassemble what that launch passed down, by the rule `LibEntry.run`
        # uses: a keyword argument wins over the tuner's and the heuristics'
        # values, which `constexprs` already carries merged together.
        meta = {**constexprs, **kwargs}
        names = tuple(mm_kernel.signature.parameters)[len(args) :]
        try:
            tail = tuple(args[3:]) + tuple(meta[name] for name in names)
        except KeyError:  # a parameter this helper cannot reconstruct
            return
        launch_grid = (tuple(grid(meta)) + (1, 1))[:3]
        if len(_MM_LAUNCH_CACHE) >= _MM_LAUNCH_LIMIT:
            # Least recently used, which a dict's insertion order gives once
            # every hit re-inserts its key below.
            coldest = next(iter(_MM_LAUNCH_CACHE), None)
            if coldest is not None:
                _MM_LAUNCH_CACHE.pop(coldest, None)
        _MM_LAUNCH_CACHE[key] = (kernel, launch_grid, tail)
        return

    # Re-insert on a hit so the order stays least-recently-used. Evicting by
    # insertion order instead loses the entries that matter: a training step
    # reuses a handful of large GEMMs recorded in the first iteration, while
    # packed variable-length sequences keep adding one-off shapes. Those fill
    # the cache, the large GEMMs are evicted as the oldest, and every launch
    # misses from then on. Measured end to end: the step holds at 41.4 s for
    # four iterations, then rises to 43.0-44.6 s -- the unaccelerated figure --
    # and stays there, while the same model with this op on the vendor backend
    # is flat across all eight. A dict move is a few hundred nanoseconds
    # against the 54 us launch this is saving.
    del _MM_LAUNCH_CACHE[key]
    _MM_LAUNCH_CACHE[key] = plan

    kernel, launch_grid, tail = plan
    if a.device.index == torch_device_fn.current_device():
        kernel[launch_grid](a, b, c, *tail)
    else:
        with torch_device_fn.device(a.device):
            kernel[launch_grid](a, b, c, *tail)


def _prune_mm_tiles(configs, named_args, **kwargs):
    if not configs:
        return configs
    args = {**named_args, **kwargs}
    m, n, k = args["M"], args["N"], args["K"]
    if 256 <= m <= 512 and 256 <= n <= 512 and k >= 128:
        preferred = [
            config
            for config in configs
            if (
                config.kwargs["BLOCK_M"],
                config.kwargs["BLOCK_N"],
                config.kwargs["BLOCK_K"],
            )
            == (64, 128, 128)
        ]
        if preferred:
            return preferred
    large_aligned = min(m, n) >= 1024 and m % 128 == 0 and n % 128 == 0
    # Size alone decides whether the wide tiles may compete; their M and N tails
    # are masked off by DIVISIBLE_M / DIVISIBLE_N. Requiring both sizes to be a
    # multiple of 128 excluded them from every training shape, and they are the
    # only configs that reach aclnn there. Device time per call, profiler, bf16,
    # each config forced in turn out of this file's own candidate list:
    #
    #   M x N x K          aclnn   128x128x256   best wide tile
    #   2240x1152x1076     45.5         62.0    42.7  (128x256x256)
    #   2240x1076x1152     71.3         79.5    59.4  (256x128x256)
    #   2240x5120x1152    101.0        142.8    94.2  (128x256x256)
    #   2240x1152x4608     92.6        107.6    92.4  (128x256x256)
    #
    # The relative error was identical across all eight configs, so the masked
    # tails are handled. M is a token count, so `m % 128` was rarely satisfied
    # -- 2240 % 128 == 64 while 1920 % 128 == 0 -- which is why the gap came and
    # went from step to step; N was unaligned on the 1076-wide layers.
    wide_allowed = min(m, n) >= 1024
    if not wide_allowed:
        configs = [
            config
            for config in configs
            if config.kwargs["BLOCK_M"] <= 128
            and config.kwargs["BLOCK_N"] <= 128
            and config.kwargs["BLOCK_K"] <= 256
        ]
    program_counts = {
        id(config): triton.cdiv(m, config.kwargs["BLOCK_M"])
        * triton.cdiv(n, config.kwargs["BLOCK_N"])
        for config in configs
    }
    min_programs = min(program_counts.values())
    max_programs = max(1, 2 * min_programs)
    candidates = [
        config for config in configs if program_counts[id(config)] <= max_programs
    ]
    if large_aligned:
        if k >= 1024:
            if args["A"].dtype == torch.float32:
                preferred = [
                    config
                    for config in candidates
                    if (
                        config.kwargs["BLOCK_M"],
                        config.kwargs["BLOCK_N"],
                        config.kwargs["BLOCK_K"],
                    )
                    in ((128, 256, 128), (256, 128, 128))
                ]
                if preferred:
                    return preferred
                preferred_block_k = (256,)
            elif args["A"].dtype in (torch.float16, torch.bfloat16):
                # A weight-gradient GEMM has the whole token count as its
                # reduction dimension, which makes it bound by the A/B tile
                # reads rather than by the Cube; BLOCK_K 256 is then not
                # automatically better. Device time per call against aclnn:
                #
                #   8704x5120x32768   aclnn 10.0 ms
                #                     256x128x256 29.7 ms   256x128x128 24.6 ms
                #
                # so keep both BLOCK_K values in play once K is that large.
                wide = ((128, 256, 256), (256, 128, 256))
                if k >= _HUGE_K:
                    wide += ((128, 256, 128), (256, 128, 128))
                preferred = [
                    config
                    for config in candidates
                    if (
                        config.kwargs["BLOCK_M"],
                        config.kwargs["BLOCK_N"],
                        config.kwargs["BLOCK_K"],
                    )
                    in wide
                ]
                if preferred:
                    return preferred
                preferred_block_k = (256,)
            else:
                preferred_block_k = (256,)
        else:
            preferred_block_k = (128,)
        preferred = [
            config
            for config in candidates
            if config.kwargs["BLOCK_K"] in preferred_block_k
        ]
        if preferred:
            return preferred
    return candidates


# Above this reduction size the GEMM is the weight-gradient one and the tile
# traversal order starts to matter; below it, measured on the shapes this
# backend sees, GROUP_M is worth 0~10% and is not worth extra autotuning.
_HUGE_K = 8192
# `mm()` used to hardcode GROUP_M=1, which was never the best choice on a
# weight-gradient shape. Device time per call against aclnn, best tile fixed:
#
#   M x N x K            aclnn   GROUP_M=1   best GROUP_M
#   8704x5120x32768    10.0 ms     25.2 ms    2  -> 24.2 ms (2.41x)
#   5120x4352x32768     6.2 ms     11.4 ms   16  ->  8.2 ms (1.32x)
#   4120x5120x32768     4.6 ms     12.0 ms   16  ->  7.8 ms (1.70x)
#
# The optimum moves with the shape, so let the tuner measure it -- but only
# where the spread is big enough to survive the measurement.
#
# 🔴 Offering a choice everywhere made the step 3 s *slower*. The tuner picks
# from one noisy benchmark, not from the truth: a repeat of the same candidate
# list has been seen to move 29%, while below this K the whole GROUP_M spread
# is 0~10%. Given (1, 8) on the ordinary shapes it moved 66 of 174 of them off
# the 1 they had been using, and that cost more than the weight-gradient
# shapes gained. So outside the huge-K regime GROUP_M stays at 1 -- what `mm`
# always passed -- and nothing is measured.
#
# `mm_out` used to pass 8 and now gets 1 on those shapes. Its spread there was
# measured at 0~10% with no consistent direction, and it is the aten `mm.out`
# entry rather than the GEMM path this is about.

# An N that is not a multiple of the chosen BLOCK_N leaves DIVISIBLE_N false,
# which masks every B load inside the K loop and forfeits the alignment hints.
# Padding B's row pitch and the output lets the unmasked path run instead.
# `_launch_addmm` pads for the same reason. Measured on 8192x4120x5120, summing
# every kernel one call issues: 2090 us unpadded against 1562 us padded, of
# which padding B and copying the result back are 109 us, so 528 us net. 128 is
# the smallest BLOCK_N any candidate uses, so aligning to it satisfies them all.
_PAD_N_TO = 128

_GROUP_M_CANDIDATES = (1, 2, 8, 16)
_GROUP_M_DEFAULTS = (1,)


def _with_group_m(configs, groups):
    """Return `configs` once per GROUP_M value, carrying the rest unchanged."""
    out = []
    for config in configs:
        for group_m in groups:
            out.append(
                triton.Config(
                    {**config.kwargs, "GROUP_M": group_m},
                    num_warps=config.num_warps,
                    num_stages=config.num_stages,
                    num_ctas=getattr(config, "num_ctas", 1),
                )
            )
    return out


def _prune_mm_configs(configs, named_args, **kwargs):
    """Pick the tiles, then attach the GROUP_M values worth measuring.

    Neither `mm()` nor `mm_out()` passes GROUP_M any more -- triton rejects a
    keyword argument that is also a config key -- so every config returned here
    must carry one.
    """
    args = {**named_args, **kwargs}
    tiles = _prune_mm_tiles(configs, named_args, **kwargs)
    groups = _GROUP_M_CANDIDATES if args["K"] >= _HUGE_K else _GROUP_M_DEFAULTS
    return _with_group_m(tiles, groups)


def _mm_heur_divisible_m(args):
    return args["M"] % args["BLOCK_M"] == 0


def _mm_heur_divisible_n(args):
    return args["N"] % args["BLOCK_N"] == 0


def _mm_grid(M, N, meta):
    num_tiles = triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"])
    return (min(num_tiles * meta.get("SPLIT_K", 1), _MAX_CORE_DIM),)


# Ascend's launch coreDim is uint16. Keep the grid bounded even when the
# compiler cannot automatically map logical programs onto physical cores.
_MAX_CORE_DIM = 65535


_MM_HEURISTICS = {
    **_hcu.HEURISTICS_CONFIGS["mm"],
    "DIVISIBLE_M": _mm_heur_divisible_m,
    "DIVISIBLE_N": _mm_heur_divisible_n,
}


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_mm_configs},
)
@triton.heuristics(_MM_HEURISTICS)
@triton.jit
def mm_kernel(
    A,
    B,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
):
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    # Preserve the base pointers before the per-tile computation advances them.
    A_base = A
    B_base = B
    C_base = C
    # Flatten both the output-tile and split-K dimensions. Capping only one
    # axis of a 2D grid would still allow their product to exceed coreDim.
    for task_id in range(
        tl.program_id(0), grid_m * grid_n * SPLIT_K, tl.num_programs(0)
    ):
        pid = task_id // SPLIT_K
        pid_z = task_id % SPLIT_K
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // (group_size)
        ram = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rbn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
        A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
        B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
        for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
            if EVEN_K:
                a = tl.load(A, mask=(ram < M)[:, None], other=0.0)
                b = tl.load(B, mask=(rbn < N)[None, :], other=0.0)
            else:
                k_remaining = K - k * (BLOCK_K * SPLIT_K)
                a = tl.load(
                    A,
                    mask=(rk[None, :] < k_remaining) & (ram < M)[:, None],
                    other=0.0,
                )
                b = tl.load(
                    B,
                    mask=(rk[:, None] < k_remaining) & (rbn < N)[None, :],
                    other=0.0,
                )
            if a.dtype != b.dtype:
                a = a.to(C.dtype.element_ty)
                b = b.to(C.dtype.element_ty)
            acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk
        acc = acc.to(C.dtype.element_ty)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        if SPLIT_K == 1:
            tl.store(C, acc, mask=mask)
        else:
            tl.atomic_add(C, acc, mask=mask)
        # Restore scalar pointers so the next tile starts from the same bases.
        A = A_base
        B = B_base
        C = C_base


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


@libentry()
@triton.jit
def _mm_narrow_n_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # A few output columns do not fill a Cube tile. Reduce independent
    # output elements on Vector cores, accumulating products in FP32.
    rows = tl.program_id(0) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    rk = tl.arange(0, BLOCK_K)
    row_acc = tl.zeros((BLOCK_OUT, BLOCK_K), tl.float32)
    for start in range(tl.cdiv(K, BLOCK_K)):
        kk = start * BLOCK_K + rk
        a_rows = tl.load(
            A + rows[:, None] * K + kk[None, :],
            mask=(rows < M)[:, None] & (kk < K)[None, :],
            other=0.0,
        ).to(tl.float32)
        b_vec = tl.load(B + kk, mask=kk < K, other=0.0).to(tl.float32)
        row_acc += a_rows * b_vec[None, :]
    tl.store(C + rows, tl.sum(row_acc, axis=1), mask=rows < M)


@libentry()
@triton.jit
def _mm_rectangular_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid // tl.cdiv(N, BLOCK_N) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid % tl.cdiv(N, BLOCK_N) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in tl.range(tl.cdiv(K, BLOCK_K)):
        kk = start * BLOCK_K + rk
        a = tl.load(
            A + rows[:, None] * K + kk[None, :],
            mask=(rows < M)[:, None] & (kk < K)[None, :],
            other=0.0,
        )
        b = tl.load(
            B + kk[:, None] * N + cols[None, :],
            mask=(kk < K)[:, None] & (cols < N)[None, :],
            other=0.0,
        )
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(b, "dot_pad_only_k")
        acc = tl.dot(a, b, acc, allow_tf32=False)
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc.to(C.dtype.element_ty),
        mask=(rows < M)[:, None] & (cols < N)[None, :],
    )


@triton.jit
def _mm_pack_narrow4_split_kernel(B, P, K: tl.constexpr, BK: tl.constexpr):
    base = tl.program_id(0) * BK
    idx = tl.arange(0, BK * 4)
    b = tl.load(B + base * 4 + idx).reshape((BK, 2, 2))
    b02, b13 = tl.split(b)
    b0, b2 = tl.split(b02)
    b1, b3 = tl.split(b13)
    k = base + tl.arange(0, BK)
    tl.store(P + k, b0)
    tl.store(P + K + k, b1)
    tl.store(P + 2 * K + k, b2)
    tl.store(P + 3 * K + k, b3)


@triton.jit
def _mm_pack_narrow_b_kernel(
    B, P, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr
):
    k = tl.program_id(0) * BK + tl.arange(0, BK)
    n = tl.arange(0, BN)
    b = tl.load(
        B + k[:, None] * N + n[None, :], (k < K)[:, None] & (n < N)[None, :], other=0
    )
    tl.store(
        P + n[:, None] * K + k[None, :],
        tl.trans(b),
        (n < N)[:, None] & (k < K)[None, :],
    )


@triton.jit
def _mm_pack_narrow4_b_kernel(
    B, P, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr
):
    k = tl.program_id(0) * BK + tl.arange(0, BK)
    n = tl.arange(0, BN)
    b = tl.load(
        B + k[:, None] * N + n[None, :], (k < K)[:, None] & (n < N)[None, :], other=0
    )
    # P is padded to 16 columns so the subsequent Cube load stays aligned.
    tl.store(P + n[:, None] * K + k[None, :], tl.trans(b), (k < K)[None, :])


@triton.jit
def _mm_packed_narrow_kernel(
    A,
    P,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, 16)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, 16), tl.float32)
    for s in tl.range(tl.cdiv(K, BK)):
        k = s * BK + rk
        a = tl.load(
            A + rows[:, None] * K + k[None, :],
            (rows < M)[:, None] & (k < K)[None, :],
            other=0,
        )
        bt = tl.load(
            P + cols[:, None] * K + k[None, :],
            (cols < N)[:, None] & (k < K)[None, :],
            other=0,
        )
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(bt, "dot_pad_only_k")
        acc = tl.dot(a, tl.trans(bt), acc, allow_tf32=False)
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc.to(C.dtype.element_ty),
        (rows < M)[:, None] & (cols < N)[None, :],
    )


@triton.jit
def _mm_packed_narrow4_kernel(
    A,
    P,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, 16)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, 16), tl.float32)
    for s in tl.range(tl.cdiv(K, BK)):
        k = s * BK + rk
        a = tl.load(
            A + rows[:, None] * K + k[None, :],
            (rows < M)[:, None] & (k < K)[None, :],
            other=0,
        )
        bt = tl.load(
            P + cols[:, None] * K + k[None, :],
            (cols < N)[:, None] & (k < K)[None, :],
            other=0,
        )
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(bt, "dot_pad_only_k")
        acc = tl.dot(a, tl.trans(bt), acc, allow_tf32=False)
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc.to(C.dtype.element_ty),
        (rows < M)[:, None] & (cols < N)[None, :],
    )


@triton.jit
def _mm_materialize_broadcast_kernel(
    X,
    Y,
    R: tl.constexpr,
    C: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    val = tl.load(X + (idx // C) * S0 + (idx % C) * S1, idx < R * C, other=0)
    tl.store(Y + idx, val, idx < R * C)


def _mm_materialize_broadcast(x):
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    with torch_device_fn.device(x.device):
        _mm_materialize_broadcast_kernel[(triton.cdiv(x.numel(), 1024),)](
            x,
            y,
            x.shape[0],
            x.shape[1],
            x.stride(0),
            x.stride(1),
            1024,
            multibuffer=False,
        )
    return y


@triton.jit
def _mm_pack_wide_panels_kernel(
    B, P, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    pid = tl.program_id(0)
    panel = pid // (K // BK)
    block_k = pid % (K // BK)
    k = block_k * BK + tl.arange(0, BK)
    n = tl.arange(0, BN)
    b = tl.load(B + k[:, None] * N + panel * BN + n[None, :])
    tl.store(P + panel * K * BN + k[:, None] * BN + n[None, :], b)


@triton.jit
def _mm_wide_panel_cube_kernel(
    A,
    P,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    panel = pid % (N // BN)
    rows = (pid // (N // BN)) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(K // BK):
        k = step * BK + rk
        a = tl.load(A + rows[:, None] * K + k[None, :], rows[:, None] < M, other=0)
        b = tl.load(P + panel * K * BN + k[:, None] * BN + cols[None, :])
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(b, "dot_pad_only_k")
        acc = tl.dot(a, b, acc, allow_tf32=False)
    tl.store(
        C + rows[:, None] * N + panel * BN + cols[None, :],
        acc.to(C.dtype.element_ty),
        rows[:, None] < M,
    )


@triton.jit
def _mm_pack_large_panels_kernel(
    B, P, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    pid = tl.program_id(0)
    panel = pid // (K // BK)
    block_k = pid % (K // BK)
    k = block_k * BK + tl.arange(0, BK)
    n = tl.arange(0, BN)
    b = tl.load(
        B + k[:, None] * N + panel * BN + n[None, :],
        (panel * BN + n[None, :]) < N,
        other=0,
    )
    tl.store(P + panel * K * BN + k[:, None] * BN + n[None, :], b)


@triton.jit
def _mm_large_panel_cube_kernel(
    A,
    P,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    panel = pid % tl.cdiv(N, BN)
    rows = (pid // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(K // BK):
        k = step * BK + rk
        a = tl.load(A + rows[:, None] * K + k[None, :], rows[:, None] < M, other=0)
        b = tl.load(P + panel * K * BN + k[:, None] * BN + cols[None, :])
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(b, "dot_pad_only_k")
        acc = tl.dot(a, b, acc, allow_tf32=False)
    tl.store(
        C + rows[:, None] * N + panel * BN + cols[None, :],
        acc.to(C.dtype.element_ty),
        (rows[:, None] < M) & (panel * BN + cols[None, :] < N),
    )


@triton.jit
def _mm_persistent_smallk_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    for step in range(4):
        pid = tl.program_id(0) + step * 16
        rows = pid // tl.cdiv(N, BLOCK_N) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid % tl.cdiv(N, BLOCK_N) * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for start in tl.range(tl.cdiv(K, BLOCK_K)):
            kk = start * BLOCK_K + rk
            a = tl.load(
                A + rows[:, None] * K + kk[None, :],
                mask=(rows < M)[:, None] & (kk < K)[None, :],
                other=0.0,
            )
            b = tl.load(
                B + kk[:, None] * N + cols[None, :],
                mask=(kk < K)[:, None] & (cols < N)[None, :],
                other=0.0,
            )
            al.compile_hint(a, "dot_pad_only_k")
            al.compile_hint(b, "dot_pad_only_k")
            acc = tl.dot(a, b, acc, allow_tf32=False)
        tl.store(
            C + rows[:, None] * N + cols[None, :],
            acc.to(C.dtype.element_ty),
            mask=(rows < M)[:, None] & (cols < N)[None, :],
        )


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def mm(a, b):
    logger.debug("GEMS_ASCEND MM")
    device = a.device
    # Both `.contiguous()` and `_mm_materialize_broadcast` preserve dtype, so
    # this holds for the whole function. The shape guards below run on every
    # launch, and this path's cost is dominated by host-side dispatch.
    both_bf16 = a.dtype == b.dtype == torch.bfloat16
    # Materialize broadcast operands to avoid a zero-stride Cube lowering failure.
    if both_bf16:
        if 0 in a.stride():
            a = _mm_materialize_broadcast(a)
        if 0 in b.stride():
            b = _mm_materialize_broadcast(b)
    # Materialize strided operands before entering the Cube kernels. This
    # avoids backend-specific failures for column-major test inputs.
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # Pack the narrow right operand once so Cube reads contiguous K rows.
    if both_bf16 and N == 16 and 2048 <= K <= 4096 and K % 1024 == 0:
        block_m = 64 if M >= 4096 else 16
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        packed_b = torch.empty((N, K), device=device, dtype=b.dtype)
        with torch_device_fn.device(device), _blockified():
            _mm_pack_narrow_b_kernel[(triton.cdiv(K, 512),)](
                b,
                packed_b,
                N,
                K,
                512,
                16,
                multibuffer=False,
            )
            _mm_packed_narrow_kernel[(triton.cdiv(M, block_m),)](
                a,
                packed_b,
                c,
                M,
                N,
                K,
                block_m,
                1024,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c
    # Deinterleave four BF16 columns into contiguous K vectors for Cube.
    if both_bf16 and N == 4 and 8192 < K <= 16384 and K % 512 == 0:
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        packed_b = torch.empty((4, K), device=device, dtype=b.dtype)
        with torch_device_fn.device(device), _blockified():
            _mm_pack_narrow4_split_kernel[(K // 512,)](
                b,
                packed_b,
                K,
                512,
                multibuffer=False,
            )
            _mm_packed_narrow_kernel[(triton.cdiv(M, 16),)](
                a,
                packed_b,
                c,
                M,
                N,
                K,
                16,
                512,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c
    # For the very wide BF16 layout, pack B into contiguous N panels so that
    # Cube reads short contiguous rows instead of the original K-strided rows.
    if (
        both_bf16
        and N >= 131072
        and N % 256 == 0
        and 1024 < K <= 2048
        and K % 128 == 0
        and 0 < M <= 128
    ):
        block_m = 16 if M <= 16 else 128
        block_n = 256
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        packed_b = torch.empty((N // block_n, K, block_n), device=device, dtype=b.dtype)
        with torch_device_fn.device(device), _blockified():
            _mm_pack_wide_panels_kernel[(N // block_n * (K // 64),)](
                b, packed_b, N, K, block_n, 64, multibuffer=False
            )
            _mm_wide_panel_cube_kernel[(triton.cdiv(M, block_m) * (N // block_n),)](
                a,
                packed_b,
                c,
                M,
                N,
                K,
                block_m,
                block_n,
                128,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c
    # Panel families: tall/narrow outputs and wide outputs with a large
    # reduction. K alignment is required by the unmasked panel kernels.
    if (
        both_bf16
        and K % 256 == 0
        and (
            (15360 < M <= 16384 and 2048 < N <= 2304 and 4096 < K <= 8192)
            or (1024 < M <= 2048 and N >= 131072 and 1024 < K <= 2048)
            or (
                1024 < M <= 2048
                and 8192 < N <= 32768
                and N % 256 != 0
                and 2048 < K <= 8192
            )
        )
    ):
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        packed_b = torch.empty(
            (triton.cdiv(N, 256), K, 256), device=device, dtype=b.dtype
        )
        with torch_device_fn.device(device), _blockified():
            _mm_pack_large_panels_kernel[(triton.cdiv(N, 256) * (K // 64),)](
                b,
                packed_b,
                N,
                K,
                256,
                64,
                multibuffer=False,
            )
            _mm_large_panel_cube_kernel[(triton.cdiv(M, 128) * triton.cdiv(N, 256),)](
                a,
                packed_b,
                c,
                M,
                N,
                K,
                128,
                256,
                256,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c
    # Compatibility exceptions: nearby shapes use the generic path in the
    # established workload. Keep their dispatch until wider ranges are measured.
    # Four M tiles per program keep this 64-tile workload balanced over 16 cores.
    if both_bf16 and (M, N, K) == (480, 4096, 128):
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        with torch_device_fn.device(device), _blockified():
            _mm_persistent_smallk_kernel[(16,)](
                a,
                b,
                c,
                M,
                N,
                K,
                128,
                256,
                128,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c
    # Wider N tiles reduce program count and K-loop overhead on these layouts.
    if both_bf16 and (
        (6144 < N <= 7168 and K == 256 and M > 128)
        or (M, N, K)
        in (
            (200, 7168, 2304),
            (13422, 256, 2048),
            (480, 4608, 7168),
        )
    ):
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        with torch_device_fn.device(device), _blockified():
            _mm_rectangular_kernel[(triton.cdiv(M, 128) * triton.cdiv(N, 256),)](
                a,
                b,
                c,
                M,
                N,
                K,
                128,
                256,
                min(256, K),
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c
    # Explicit tiles for narrow outputs and the remaining slow rectangular layouts.
    if both_bf16 and (M, N, K) in (
        (1, 64, 4096),
        (6, 64, 4096),
        (16, 64, 2048),
        (4434, 2048, 1024),
        (1352, 3072, 2048),
        (80, 12288, 2048),
        (368, 3072, 2048),
    ):
        bm = 16 if M <= 16 else 128
        bn = 64 if N == 64 else 256
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        with torch_device_fn.device(device), _blockified():
            _mm_rectangular_kernel[(triton.cdiv(M, bm) * triton.cdiv(N, bn),)](
                a,
                b,
                c,
                M,
                N,
                K,
                bm,
                bn,
                256,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    dot_out_dtype = tl.float32

    # Keep tiny reductions off the autotuned Cube path. Some Ascend
    # toolchains select an invalid tile for these partial shapes.
    if M <= 16 and N <= 16 and K < 128:
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        block_k = min(128, triton.next_power_of_2(K))
        with torch_device_fn.device(device), _blockified():
            _mm_rectangular_kernel[(1,)](
                a,
                b,
                c,
                M,
                N,
                K,
                16,
                16,
                block_k,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c

    # Keep all short reductions off the autotuned Cube path. Some Ascend
    # toolchains select an invalid tile for partial K tiles.
    if K < 128:
        block_m = 16 if M <= 16 else 128
        block_n = 32 if N <= 32 else (128 if N <= 128 else 256)
        with torch_device_fn.device(device), _blockified():
            _mm_rectangular_kernel[
                (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
            ](
                a,
                b,
                c,
                M,
                N,
                K,
                block_m,
                block_n,
                triton.next_power_of_2(K),
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c

    if a.dtype == b.dtype and a.dtype in (torch.float16, torch.bfloat16) and K >= 128:
        if N == 1:
            with torch_device_fn.device(device):
                _mm_narrow_n_kernel[(triton.cdiv(M, 4),)](
                    a,
                    b,
                    c,
                    M,
                    N,
                    K,
                    BLOCK_OUT=4,
                    BLOCK_K=min(2048, triton.next_power_of_2(K)),
                )
            return c
        if M <= 32 and N >= 16384:
            block_m = 16 if M <= 16 else 32
            block_n = 256
            with torch_device_fn.device(device), _blockified():
                _mm_rectangular_kernel[
                    (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
                ](
                    a,
                    b,
                    c,
                    M,
                    N,
                    K,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=256,
                    multibuffer=True,
                    unit_flag=True,
                    sync_solver=False,
                )
            return c

    # launch kernel
    assert dot_out_dtype is _MM_KWARGS["dot_out_dtype"]
    # Pad N only where the masked path costs more than the two copies: the
    # reduction has to be long enough for the K loop's masking to dominate, and
    # the output wide enough that one extra row of tiles is a small share.
    pad_n = 0
    if N % _PAD_N_TO != 0 and M >= 4096 and N >= 1024 and K >= 1024:
        pad_n = triton.cdiv(N, _PAD_N_TO) * _PAD_N_TO
    if pad_n:
        b_work = torch.empty((K, pad_n), device=device, dtype=b.dtype)
        b_work[:, :N].copy_(b)
        # The kernel reads the padding unmasked and writes it to columns the
        # caller never sees, but leaving it uninitialized would let a stray NaN
        # through tl.dot; zeroing the strip is cheap and keeps it deterministic.
        b_work[:, N:].zero_()
        c_work = torch.empty((M, pad_n), device=device, dtype=c_dtype)
    else:
        b_work, c_work = b, c
    launch_n = pad_n or N
    _launch_mm_kernel(
        (
            a,
            b_work,
            c_work,
            M,
            launch_n,
            K,
            a.stride(0),
            a.stride(1),
            b_work.stride(0),
            b_work.stride(1),
            c_work.stride(0),
            c_work.stride(1),
        ),
        _MM_KWARGS,
        "mm",
        lambda META: _mm_grid(M, launch_n, META),
    )
    if pad_n:
        # Hand back a contiguous (M, N): a view into the padded buffer would be
        # non-contiguous, and callers of `mm` do not expect that.
        c.copy_(c_work[:, :N])
    return c


def mm_out(a, b, *, out):
    logger.debug("GEMS_ASCEND MM_OUT")
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    _launch_mm_kernel(
        (
            a,
            b,
            out,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
        ),
        _MM_OUT_KWARGS,
        "mm_out",
        lambda META: _mm_grid(M, N, META),
    )
    return out
