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

import importlib
import random

import numpy as np
import pytest
import torch
import triton

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    MNK_SHAPES = [
        (1, 1, 32),
    ]
    FLOAT_DTYPES = [torch.float32]
else:
    MNK_SHAPES = [
        (1, 1, 32),
        (15, 160, 1024),
        (495, 5333, 71),
    ]
    FLOAT_DTYPES = utils.FLOAT_DTYPES


MK_SHAPES = (
    [(1, 32)]
    if QUICK_MODE
    else [
        (1, 32),
        (7, 33),
        (31, 65),
        (160, 1024),
        (257, 96),
        (1023, 255),
        (5333, 71),
    ]
)


def _mm_atol_base():
    """On MetaX C550, the hardware matmul unit has inherent precision limits in
    fp16/bf16 (confirmed by native torch.mm showing the same error vs fp64 ref).
    Use a larger atol base to accommodate hardware accumulation precision."""
    if flag_gems.vendor_name == "metax":
        return 3e-4
    return 1e-4


# Issue #2833: fails at (1, 1, 2)
@pytest.mark.mm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("b_column_major", [True, False])
def test_mm(M, N, K, dtype, b_column_major):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 mm test on tsingmicro platform")

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    if b_column_major:
        mat2 = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    else:
        mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    res_out = flag_gems.mm(mat1, mat2)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K, atol=_mm_atol_base())


@pytest.mark.mm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_broadcast_stride_zero(dtype):
    """Regression test: broadcast tensors (stride=0) must not crash TMA path."""
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #3794: not working ")
    torch.manual_seed(0)
    M, K, N = 128, 256, 256

    # Simulate the stride=(0,0) tensor that autograd produces from sum().backward():
    # scalar expand -> all strides are 0
    a = torch.randn((), dtype=dtype, device=flag_gems.device).expand(M, K)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    assert a.stride() == (0, 0)

    ref_a = utils.to_reference(a.contiguous(), True)
    ref_b = utils.to_reference(b, True)

    ref_out = torch.mm(ref_a, ref_b)
    res_out = flag_gems.mm(a, b)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K, atol=_mm_atol_base())


@pytest.mark.mm
def test_mm_out_vllm_tma_column_major_weight():
    """Regression test for vLLM Inductor mm_out with a column-major BF16 weight."""
    torch.manual_seed(0)
    M, K, N = 4096, 4096, 3328
    dtype = torch.bfloat16

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2_storage = torch.randn((N, K), dtype=dtype, device=flag_gems.device)
    mat2 = mat2_storage.t()
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)

    assert mat2.shape == (K, N)
    assert mat2.stride() == (1, K)

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_out = torch.empty((M, N), dtype=ref_mat1.dtype, device=ref_mat1.device)
    torch.mm(ref_mat1, ref_mat2, out=ref_out)

    flag_gems.mm_out(mat1, mat2, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K, atol=_mm_atol_base())


@pytest.mark.mm
@pytest.mark.skipif(
    not hasattr(
        getattr(getattr(triton, "tools", None), "tensor_descriptor", None),
        "TensorDescriptor",
    )
    or flag_gems.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 9,
    reason="Host TMA TensorDescriptor and Hopper GPU are required for this regression test.",
)
def test_mm_kernel_general_host_tma_vllm_column_major_weight_compile_error():
    """Reproduce the vLLM TMA descriptor compile error for a column-major BF16 weight."""
    from triton.tools.tensor_descriptor import TensorDescriptor

    from flag_gems.runtime.backend._nvidia.hopper.ops.mm import (
        mm_kernel_general_host_tma,
    )

    torch.manual_seed(0)
    M, K, N = 64, 4096, 3328
    dtype = torch.bfloat16

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2_storage = torch.randn((N, K), dtype=dtype, device=flag_gems.device)
    mat2 = mat2_storage.t()
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)

    assert mat2.shape == (K, N)
    assert mat2.stride() == (1, K)

    dummy_block = [1, 1]
    a_desc = TensorDescriptor(mat1, mat1.shape, mat1.stride(), dummy_block)
    b_desc = TensorDescriptor(mat2, mat2.T.shape, mat2.T.stride(), dummy_block)
    c_desc = TensorDescriptor(out, out.shape, out.stride(), dummy_block)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    mm_kernel_general_host_tma.fn.fn[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        mat1.stride(0),
        mat1.stride(1),
        mat2.stride(0),
        mat2.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=64,
        GROUP_M=8,
        A_ROW_MAJOR=True,
        B_ROW_MAJOR=False,
        dtype="bfloat16",
        num_warps=4,
        num_stages=2,
    )


@pytest.mark.mm
@pytest.mark.mm_self_transpose
@pytest.mark.parametrize("M, K", MK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_self_transpose(M, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip(
            "Issue #2834: Skipping fp32 mm self-transpose test on tsingmicro platform"
        )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    mat = torch.randn((K, M), dtype=dtype, device=flag_gems.device).t()
    ref_mat = utils.to_reference(mat, True)

    ref_out = torch.mm(ref_mat, ref_mat.t())
    res_out = flag_gems.mm(mat, mat.t())

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K, atol=_mm_atol_base())


@pytest.mark.mm_out
@pytest.mark.parametrize("M, K", MK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_out_self_transpose(M, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip(
            "Issue #2834: Skipping fp32 mm.out self-transpose test on tsingmicro platform"
        )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    mat = torch.randn((K, M), dtype=dtype, device=flag_gems.device).t()
    out = torch.empty((M, M), dtype=dtype, device=flag_gems.device)
    ref_mat = utils.to_reference(mat, True)
    ref_out = utils.to_reference(out, True)

    torch.mm(ref_mat, ref_mat.t(), out=ref_out)
    flag_gems.mm_out(mat, mat.t(), out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K, atol=_mm_atol_base())


# Public mm and mm_out accuracy regressions on Ascend.
@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend mm accuracy regression"
)
@pytest.mark.parametrize(
    "M, N, K",
    [
        (4080, 8224, 32),
        (4096, 8192, 33),
        (4352, 10240, 64),
        (4096, 4096, 65),
    ],
)
def test_mm_ascend_large_shapes(M, N, K):
    """Check public mm on large outputs with short reductions."""
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.float16, device=flag_gems.device)
    b = torch.randn((K, N), dtype=torch.float16, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a).float(), utils.to_reference(b).float())
    out = flag_gems.mm(a, b)
    utils.gems_assert_close(out, ref, a.dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend mm accuracy regression"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("K", [0, 33, 64, 71, 129])
def test_mm_ascend_transposed_inputs_and_k_tails(dtype, K):
    """Cover transposed inputs, irregular dimensions and an empty reduction."""
    torch.manual_seed(0)
    M, N = 145, 97
    a = torch.randn((K, M), dtype=dtype, device=flag_gems.device).t()
    b = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    ref = torch.mm(utils.to_reference(a).float(), utils.to_reference(b).float())
    out = flag_gems.mm(a, b)
    utils.gems_assert_close(out, ref, dtype, reduce_dim=max(K, 1))


@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend mm accuracy regression"
)
@pytest.mark.parametrize("use_out", [False, True])
def test_mm_ascend_large_shape_entrypoints(use_out):
    """Check public mm and mm_out with a large output and transposed weight."""
    torch.manual_seed(0)
    M, N, K = 4352, 10240, 33
    a = torch.randn((M, K), dtype=torch.bfloat16, device=flag_gems.device)
    b = torch.randn((N, K), dtype=torch.bfloat16, device=flag_gems.device).t()
    ref = torch.mm(utils.to_reference(a).float(), utils.to_reference(b).float())
    if use_out:
        out = torch.empty((M, N), dtype=a.dtype, device=a.device)
        assert flag_gems.mm_out(a, b, out=out) is out
    else:
        out = flag_gems.mm(a, b)
    utils.gems_assert_close(out, ref, a.dtype, reduce_dim=K)


# Ascend launch-plan caching, N padding and the bfloat16 strided fast paths.
@pytest.fixture(scope="module")
def ascend_mm():
    return importlib.import_module("flag_gems.runtime.backend._ascend.ops.mm")


@pytest.fixture
def ascend_mm_caches(ascend_mm, monkeypatch):
    # Both caches live for the life of the process. Bind empty containers for
    # the test instead of clearing and refilling the real ones: the launch
    # cache holds compiled kernels, and putting entries back after the test
    # has recompiled under them leaves handles on stale artifacts.
    plans_type = type(ascend_mm._MM_TN_PLANS)
    monkeypatch.setattr(ascend_mm, "_MM_LAUNCH_CACHE", {})
    monkeypatch.setattr(ascend_mm, "_MM_TN_PLANS", plans_type())
    return ascend_mm


def _mm_strided_operands(layout, M, N, K, dtype=torch.bfloat16):
    # Strides, not shapes, select which fast path the gate admits.
    device = flag_gems.device
    if layout == "nt":
        a = torch.randn((M, K), dtype=dtype, device=device)
        b = torch.randn((N, K), dtype=dtype, device=device).t()
    elif layout == "nn":
        a = torch.randn((M, K), dtype=dtype, device=device)
        b = torch.randn((K, N), dtype=dtype, device=device)
    else:
        a = torch.randn((K, M), dtype=dtype, device=device).t()
        b = torch.randn((K, N), dtype=dtype, device=device)
    return a, b


def _mm_strided_plan_key(a, b, use_out):
    # Mirrors the key _try_mm_tn builds. Seeding a plan under it pins the
    # config, bypassing the runtime search, which times three samples per
    # candidate and may pick a different config on each cold run.
    M, K = a.shape
    N = b.shape[1]
    key = (a.device.index, M, N, K, use_out)
    if a.is_contiguous() and b.stride() == (1, K):
        return key + ("nt",)
    if a.is_contiguous() and b.is_contiguous():
        return key + ("nn",)
    return key


@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend mm launch-plan cache"
)
def test_mm_ascend_launch_cache_reuses_the_plan(ascend_mm_caches):
    mod = ascend_mm_caches
    torch.manual_seed(0)
    M, N, K = 256, 256, 256
    a = torch.randn((M, K), dtype=torch.float16, device=flag_gems.device)
    b = torch.randn((K, N), dtype=torch.float16, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a).float(), utils.to_reference(b).float())

    first = mod.mm(a, b)
    recorded = list(mod._MM_LAUNCH_CACHE)
    assert recorded, "the first launch should record a plan"

    second = mod.mm(a, b)
    assert list(mod._MM_LAUNCH_CACHE) == recorded, "a repeat must not add a key"

    utils.gems_assert_close(first, ref, a.dtype, reduce_dim=K)
    utils.gems_assert_close(second, ref, a.dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend mm launch-plan cache"
)
def test_mm_ascend_launch_cache_evicts_the_least_recently_used(
    ascend_mm_caches, monkeypatch
):
    # A training step reuses a few large GEMMs recorded in the first iteration
    # while one-off shapes keep arriving. Evicting by insertion order drops the
    # reused ones; eviction has to follow last use instead.
    mod = ascend_mm_caches
    monkeypatch.setattr(mod, "_MM_LAUNCH_LIMIT", 2)
    torch.manual_seed(0)
    operands = {
        n: (
            torch.randn((256, 256), dtype=torch.float16, device=flag_gems.device),
            torch.randn((256, n), dtype=torch.float16, device=flag_gems.device),
        )
        for n in (256, 384, 512)
    }

    mod.mm(*operands[256])
    mod.mm(*operands[384])
    recorded = list(mod._MM_LAUNCH_CACHE)
    if len(recorded) != 2:
        pytest.skip("these shapes do not each record exactly one launch plan")

    mod.mm(*operands[256])
    assert list(mod._MM_LAUNCH_CACHE) == recorded[::-1], "a hit must move to the end"

    mod.mm(*operands[512])
    assert recorded[0] in mod._MM_LAUNCH_CACHE, "the recently used key was evicted"
    assert recorded[1] not in mod._MM_LAUNCH_CACHE, "the coldest key survived"


@pytest.mark.mm
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend mm N padding")
@pytest.mark.parametrize(
    "N, K, expect_pad",
    [
        (1025, 1024, False),  # N*K = 1.0M: the 127 extra columns dominate
        (2049, 2048, True),  # N*K = 4.2M: the masked tail dominates instead
    ],
)
def test_mm_ascend_pads_n_only_above_the_size_product(
    ascend_mm_caches, N, K, expect_pad
):
    mod = ascend_mm_caches
    torch.manual_seed(0)
    M = 4096
    a = torch.randn((M, K), dtype=torch.float16, device=flag_gems.device)
    b = torch.randn((K, N), dtype=torch.float16, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a).float(), utils.to_reference(b).float())

    out = mod.mm(a, b)
    utils.gems_assert_close(out, ref, a.dtype, reduce_dim=K)

    launched_n = {key[9] for key in mod._MM_LAUNCH_CACHE if len(key) > 9}
    padded = triton.cdiv(N, mod._PAD_N_TO) * mod._PAD_N_TO
    assert N * K >= mod._PAD_MIN_NK if expect_pad else N * K < mod._PAD_MIN_NK
    if expect_pad:
        assert (
            padded in launched_n
        ), f"expected a launch at N={padded}, saw {launched_n}"
    else:
        assert padded not in launched_n, f"padded below the product floor: {launched_n}"


@pytest.mark.mm
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend mm strided paths")
def test_mm_ascend_plan_rank_orders_ties_without_the_clock():
    rank = importlib.import_module(
        "flag_gems.runtime.backend._ascend.ops.mm"
    )._mm_tn_plan_rank
    # The default sorts ahead of every candidate.
    assert rank(None) < rank((128, 128, 256, 2, 1))
    # A smaller split sorts ahead, since split-K reorders the reduction.
    assert rank((128, 128, 256, 2, 1)) < rank((128, 128, 256, 2, 2))
    assert rank((128, 128, 256, 2, 2)) < rank((128, 128, 256, 2, 4))
    # Remaining fields only break the last ties, and do so deterministically.
    assert rank((128, 256, 256, 8, 1)) < rank((256, 128, 256, 8, 1))


@pytest.mark.mm
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend mm strided paths")
def test_mm_ascend_tie_band_keeps_the_default_when_nothing_is_faster(
    ascend_mm_caches, monkeypatch
):
    # With every plan timing the same, only the tie rule decides, and it has to
    # keep the default: three timed rounds cannot separate plans a few percent
    # apart, and letting them decide would vary between processes.
    mod = ascend_mm_caches
    torch.manual_seed(0)
    a, b = _mm_strided_operands("tn", 256, 256, 8192)

    class _FlatEvent:
        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 1.0

    monkeypatch.setattr(
        mod.torch_device_fn, "Event", lambda *args, **kwargs: _FlatEvent()
    )
    assert mod._try_mm_tn(a, b) is None
    assert mod._MM_TN_PLANS, "the decision should still be cached"
    assert next(iter(mod._MM_TN_PLANS.values())) is None


@pytest.mark.mm
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend mm strided gate")
@pytest.mark.parametrize(
    "case",
    ["float16", "short_k", "unaligned_m", "unaligned_n", "batched", "aliased_out"],
)
def test_mm_ascend_strided_gate_rejects(ascend_mm_caches, case):
    # Every rejection has to leave the caller on the default implementation.
    mod = ascend_mm_caches
    torch.manual_seed(0)
    device = flag_gems.device
    out = None
    if case == "float16":
        a, b = _mm_strided_operands("tn", 256, 256, 8192, dtype=torch.float16)
    elif case == "short_k":
        a, b = _mm_strided_operands("tn", 256, 256, 64)
    elif case == "unaligned_m":
        a, b = _mm_strided_operands("tn", 248, 256, 8192)
    elif case == "unaligned_n":
        a, b = _mm_strided_operands("tn", 256, 248, 8192)
    elif case == "batched":
        a = torch.randn((2, 256, 8192), dtype=torch.bfloat16, device=device)
        b = torch.randn((2, 8192, 256), dtype=torch.bfloat16, device=device)
    else:
        a, b = _mm_strided_operands("tn", 256, 256, 8192)
        # An output aliasing an input would be overwritten while still read.
        out = b.view(-1)[: 256 * 256].view(256, 256)

    assert mod._try_mm_tn(a, b, out) is None


@pytest.mark.mm
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend mm strided paths")
@pytest.mark.parametrize(
    "layout, M, N, K, config",
    [
        # nt and nn reach _mm_strided_direct_kernel; tn reaches
        # _mm_tn_split_kernel, and its split>1 case the partial reduction.
        ("nt", 256, 2048, 4096, (128, 256, 256, 2, 1)),
        ("nn", 256, 512, 1152, (128, 256, 128, 2, 1)),
        ("tn", 256, 256, 8192, (128, 128, 256, 2, 1)),
        ("tn", 256, 256, 16384, (128, 128, 128, 2, 2)),
    ],
)
@pytest.mark.parametrize("use_out", [False, True])
def test_mm_ascend_strided_matches_the_default_path(
    ascend_mm_caches, layout, M, N, K, config, use_out
):
    mod = ascend_mm_caches
    torch.manual_seed(0)
    a, b = _mm_strided_operands(layout, M, N, K)
    mod._MM_TN_PLANS[_mm_strided_plan_key(a, b, use_out)] = config

    if use_out:
        out = torch.empty((M, N), dtype=a.dtype, device=a.device)
        fast = mod._try_mm_tn(a, b, out)
        assert fast is out
        slow = mod._mm_out_default(
            a, b, out=torch.empty((M, N), dtype=a.dtype, device=a.device)
        )
    else:
        fast = mod._try_mm_tn(a, b)
        assert fast is not None
        slow = mod._mm_default(a, b)

    ref = torch.mm(utils.to_reference(a).float(), utils.to_reference(b).float())
    utils.gems_assert_close(fast, ref, a.dtype, reduce_dim=K)

    if config[4] == 1:
        # One K pass accumulates in the same order as the default, and both
        # accumulate in FP32, so the bfloat16 results agree exactly.
        assert torch.equal(fast, slow), "fast path diverged from the default"
    else:
        # Splitting K sums per-split FP32 partials, which reorders the
        # reduction, so the last bit may differ. The default is itself a
        # bfloat16 result rather than ground truth, so compare both against
        # the FP32 reference and require the fast path to be no worse.
        # Route through to_reference so this holds under --ref=cpu too.
        fast_err = (utils.to_reference(fast).float() - ref).abs().max().item()
        slow_err = (utils.to_reference(slow).float() - ref).abs().max().item()
        assert fast_err <= 2 * slow_err + 1e-6, (
            f"split-K is less accurate than the default: "
            f"{fast_err:.6g} vs {slow_err:.6g}"
        )


@pytest.mark.mm
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend mm strided paths")
def test_mm_ascend_strided_plan_cache_separates_out_from_allocated(
    ascend_mm_caches, monkeypatch
):
    # The allocated-output and caller-owned-output paths have different
    # baselines, so a decision made for one must not stand in for the other.
    mod = ascend_mm_caches
    torch.manual_seed(0)
    M, N, K = 256, 256, 8192
    a, b = _mm_strided_operands("tn", M, N, K)
    seeded = _mm_strided_plan_key(a, b, False)
    mod._MM_TN_PLANS[seeded] = (128, 128, 256, 2, 1)

    searches = []

    def decline(*args, **kwargs):
        searches.append(args)
        return None

    monkeypatch.setattr(mod, "_choose_mm_tn_plan", decline)

    assert mod._try_mm_tn(a, b) is not None, "the seeded plan was not reused"
    assert not searches, "a cached key must not trigger a search"

    out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    assert mod._try_mm_tn(a, b, out) is None, "the stub declines every candidate"
    assert len(searches) == 1, "the out variant has its own key and must search"
    assert _mm_strided_plan_key(a, b, True) in mod._MM_TN_PLANS
