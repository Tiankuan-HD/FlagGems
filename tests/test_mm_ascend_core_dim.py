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

import pytest
import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.utils.libentry import LibEntry

from . import accuracy_utils as utils

pytestmark = [
    pytest.mark.mm,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend", reason="Ascend launch-grid regression"
    ),
]


@pytest.fixture(scope="module")
def ascend_mm():
    return importlib.import_module("flag_gems.runtime.backend._ascend.ops.mm")


def _run_kernel(ascend_mm, a, b, out, split_k):
    # Exercise the smallest tuning candidate directly: autotuning may select
    # a larger tile and otherwise hide the overflowing-grid regression.
    kernel = ascend_mm.mm_kernel
    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn
    M, K = a.shape
    N = b.shape[1]
    meta = {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": split_k}
    grid = ascend_mm._mm_grid(M, N, meta)
    assert len(grid) == 1 and 0 < grid[0] <= 65535
    kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        *a.stride(),
        *b.stride(),
        *out.stride(),
        dot_out_dtype=tl.float32,
        GROUP_M=8,
        EVEN_K=K % (meta["BLOCK_K"] * split_k) == 0,
        **meta,
    )


@pytest.mark.parametrize(
    "M, N, K, split_k",
    [
        (4080, 8224, 32, 1),  # Exactly 65535 logical programs.
        (4096, 8192, 33, 1),  # 65536 programs; one tile after the first pass.
        (4352, 10240, 64, 1),  # 87040 programs, matching the reported coreDim.
        (4096, 4096, 65, 2),  # The split-K factor alone crosses the limit.
    ],
)
def test_mm_ascend_core_dim_boundary(ascend_mm, M, N, K, split_k):
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.float16, device=flag_gems.device)
    b = torch.randn((K, N), dtype=torch.float16, device=flag_gems.device)
    # Use FP32 for split-K to avoid rounding each partial sum to FP16 before
    # the atomic add; this test checks launch limits and tile coverage.
    out_dtype = torch.float32 if split_k > 1 else a.dtype
    out = torch.zeros((M, N), dtype=out_dtype, device=flag_gems.device)
    ref = torch.mm(a.float(), b.float())
    _run_kernel(ascend_mm, a, b, out, split_k)
    utils.gems_assert_close(out, ref, out.dtype, reduce_dim=K)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("split_k, K", [(1, 0), (1, 64), (1, 71), (2, 33), (2, 129)])
def test_mm_ascend_repeated_tiles(ascend_mm, monkeypatch, dtype, split_k, K):
    # Force many iterations with small allocations; also cover a partial
    # GROUP_M, masked M/N/K tails and transposed inputs.
    monkeypatch.setattr(ascend_mm, "_MAX_CORE_DIM", 3)
    torch.manual_seed(0)
    M, N = 145, 97
    a = torch.randn((K, M), dtype=dtype, device=flag_gems.device).t()
    b = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    out_dtype = torch.float32 if split_k > 1 else dtype
    out = torch.zeros((M, N), dtype=out_dtype, device=flag_gems.device)
    ref = torch.mm(a.float(), b.float())
    _run_kernel(ascend_mm, a, b, out, split_k)
    utils.gems_assert_close(out, ref, out.dtype, reduce_dim=max(K, 1))


@pytest.mark.parametrize("use_out", [False, True])
def test_mm_ascend_large_grid_entrypoints(ascend_mm, use_out):
    torch.manual_seed(0)
    M, N, K = 4352, 10240, 33
    a = torch.randn((M, K), dtype=torch.bfloat16, device=flag_gems.device)
    b = torch.randn((N, K), dtype=torch.bfloat16, device=flag_gems.device).t()
    ref = torch.mm(a.float(), b.float())
    if use_out:
        out = torch.empty((M, N), dtype=a.dtype, device=a.device)
        assert ascend_mm.mm_out(a, b, out=out) is out
    else:
        out = ascend_mm.mm(a, b)
    utils.gems_assert_close(out, ref, a.dtype, reduce_dim=K)


# ---------------------------------------------------------------------------
# Launch-cache and N-padding behaviour of the same backend.
# ---------------------------------------------------------------------------


@pytest.fixture
def bound_mm():
    """The module `flag_gems.mm` is actually bound to, with an empty cache.

    Not the same object as `ascend_mm` above. The backend is importable both
    as `_ascend.ops.mm` and as `flag_gems.runtime.backend._ascend.ops.mm`, and
    those are two distinct module objects with their own module-level state.
    The cases below inspect that state, so they have to reach the one the
    operator was bound to -- the other one shows an empty cache for a launch
    that was in fact served.
    """
    mm_mod = importlib.import_module(flag_gems.mm.__module__)
    mm_mod._MM_LAUNCH_CACHE.clear()
    yield mm_mod
    mm_mod._MM_LAUNCH_CACHE.clear()


@pytest.fixture
def count_run(monkeypatch):
    """Count LibEntry.run calls: a cached launch must not reach it."""
    calls = []
    original = LibEntry.run

    def counted(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LibEntry, "run", counted)
    return calls


# A cached launch must agree with the uncached one, so every case runs the same
# inputs twice and compares the second result against the first bitwise, not
# just against a tolerance: the two go through the same compiled kernel and must
# produce identical bytes.
@pytest.mark.parametrize(
    "M, N, K",
    [
        (145, 97, 256),  # masked M and N tails
        (256, 512, 256),  # hits the preferred-tile prune branch
        (1024, 1152, 1076),  # a shape from the Qwen3.5 training mix
        (2240, 1152, 288),  # same N, different M: a separate key
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_cached_launch_matches_first(bound_mm, M, N, K, dtype):
    mm_mod = bound_mm
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a).float(), utils.to_reference(b).float())

    first = flag_gems.mm(a, b)
    filled = len(mm_mod._MM_LAUNCH_CACHE)
    second = flag_gems.mm(a, b)

    assert filled >= 1, "the first launch did not populate the cache"
    assert len(mm_mod._MM_LAUNCH_CACHE) == filled, "a repeat added a second entry"
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    utils.gems_assert_close(second, ref, dtype, reduce_dim=K)


def test_mm_cached_launch_skips_libentry(bound_mm, count_run):
    """The point of the cache: the second launch does not re-enter LibEntry."""
    a = torch.randn((512, 256), dtype=torch.bfloat16, device=flag_gems.device)
    b = torch.randn((256, 512), dtype=torch.bfloat16, device=flag_gems.device)

    flag_gems.mm(a, b)
    after_first = len(count_run)
    assert after_first >= 1, "the first launch should go through LibEntry"
    flag_gems.mm(a, b)
    assert len(count_run) == after_first, "the cached launch re-entered LibEntry"


def test_mm_interleaved_shapes_do_not_collide(bound_mm):
    """Two shapes alternating must each keep their own compiled launch."""
    mm_mod = bound_mm
    torch.manual_seed(0)
    small = (
        torch.randn((128, 256), dtype=torch.bfloat16, device=flag_gems.device),
        torch.randn((256, 128), dtype=torch.bfloat16, device=flag_gems.device),
    )
    large = (
        torch.randn((1024, 512), dtype=torch.bfloat16, device=flag_gems.device),
        torch.randn((512, 1024), dtype=torch.bfloat16, device=flag_gems.device),
    )
    refs = [
        torch.mm(utils.to_reference(x).float(), utils.to_reference(y).float())
        for x, y in (small, large)
    ]

    for _ in range(3):
        for (x, y), ref in zip((small, large), refs):
            out = flag_gems.mm(x, y)
            assert out.shape == ref.shape
            utils.gems_assert_close(out, ref, torch.bfloat16, reduce_dim=x.shape[1])
    assert len(mm_mod._MM_LAUNCH_CACHE) == 2


def test_mm_transposed_operands_keyed_separately(bound_mm):
    """A transposed operand is materialized, so its strides -- and its key --
    differ from the same logical shape passed contiguously."""
    torch.manual_seed(0)
    M, N, K = 256, 320, 256
    a = torch.randn((M, K), dtype=torch.bfloat16, device=flag_gems.device)
    b = torch.randn((K, N), dtype=torch.bfloat16, device=flag_gems.device)
    bt = torch.randn((N, K), dtype=torch.bfloat16, device=flag_gems.device).t()

    for operand in (b, bt):
        ref = torch.mm(
            utils.to_reference(a).float(), utils.to_reference(operand).float()
        )
        for _ in range(2):
            utils.gems_assert_close(
                flag_gems.mm(a, operand), ref, torch.bfloat16, reduce_dim=K
            )


def test_mm_out_and_mm_share_no_entry(bound_mm):
    """`mm` and `mm_out` compile different kernels (GROUP_M differs), so their
    entries must stay apart even for identical shapes and strides."""
    mm_mod = bound_mm
    torch.manual_seed(0)
    M, N, K = 512, 512, 256
    a = torch.randn((M, K), dtype=torch.bfloat16, device=flag_gems.device)
    b = torch.randn((K, N), dtype=torch.bfloat16, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a).float(), utils.to_reference(b).float())

    out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    for _ in range(2):
        utils.gems_assert_close(flag_gems.mm(a, b), ref, a.dtype, reduce_dim=K)
        assert mm_mod.mm_out(a, b, out=out) is out
        utils.gems_assert_close(out, ref, a.dtype, reduce_dim=K)
    assert len(mm_mod._MM_LAUNCH_CACHE) == 2


def test_launch_cache_evicts_without_losing_correctness(bound_mm):
    """A cache at its limit must still return correct results for every key."""
    mm_mod = bound_mm
    cache = mm_mod._MM_LAUNCH_CACHE
    original_limit = mm_mod._MM_LAUNCH_LIMIT
    mm_mod._MM_LAUNCH_LIMIT = 2
    try:
        torch.manual_seed(0)
        pairs = []
        for M in (128, 256, 384):
            a = torch.randn((M, 256), dtype=torch.bfloat16, device=flag_gems.device)
            b = torch.randn((256, 128), dtype=torch.bfloat16, device=flag_gems.device)
            ref = torch.mm(utils.to_reference(a).float(), utils.to_reference(b).float())
            pairs.append((a, b, ref))
        for _ in range(3):
            for a, b, ref in pairs:
                utils.gems_assert_close(
                    flag_gems.mm(a, b), ref, torch.bfloat16, reduce_dim=256
                )
        assert len(cache) <= 2
    finally:
        mm_mod._MM_LAUNCH_LIMIT = original_limit


def test_launch_cache_keeps_the_reused_entry(bound_mm, count_run):
    """Eviction must drop the least recently used entry, not the oldest one.

    A training step reuses a handful of large GEMMs while packed
    variable-length sequences keep adding one-off shapes. Evicting by
    insertion order drops the reused entries first and every launch misses
    from then on, which showed up end to end as the step time climbing back
    to the unaccelerated figure after a few iterations.

    The assertion is that the reused shape is never re-entered into
    `LibEntry.run` after its first launch. Checking whether its entry is
    present at the end does not distinguish the two policies: the last launch
    of that shape re-inserts it on a miss either way.
    """
    mm_mod = bound_mm
    original_limit = mm_mod._MM_LAUNCH_LIMIT
    mm_mod._MM_LAUNCH_LIMIT = 2
    try:
        torch.manual_seed(0)
        hot = (
            torch.randn((128, 256), dtype=torch.bfloat16, device=flag_gems.device),
            torch.randn((256, 128), dtype=torch.bfloat16, device=flag_gems.device),
        )
        hot_ref = torch.mm(
            utils.to_reference(hot[0]).float(), utils.to_reference(hot[1]).float()
        )

        flag_gems.mm(*hot)
        after_first = len(count_run)
        assert after_first >= 1, "the first launch should go through LibEntry"

        # Each one-off shape arrives after the hot one has just been used, so
        # under LRU the hot entry is never the eviction candidate.
        for m in (256, 384, 512):
            one_off = (
                torch.randn((m, 256), dtype=torch.bfloat16, device=flag_gems.device),
                torch.randn((256, 128), dtype=torch.bfloat16, device=flag_gems.device),
            )
            before = len(count_run)
            utils.gems_assert_close(
                flag_gems.mm(*hot), hot_ref, torch.bfloat16, reduce_dim=256
            )
            assert (
                len(count_run) == before
            ), f"the reused entry was evicted before the shape with M={m}"
            flag_gems.mm(*one_off)
    finally:
        mm_mod._MM_LAUNCH_LIMIT = original_limit


@pytest.mark.parametrize(
    "M, N, K, padded",
    [
        (4096, 1160, 1024, True),  # every guard satisfied: N is padded to 1280
        (4096, 1152, 1024, False),  # N is already a multiple of 128
        (2048, 1160, 1024, False),  # M below the threshold
        (4096, 1160, 512, False),  # K below the threshold
    ],
)
def test_mm_pads_an_unaligned_n(bound_mm, M, N, K, padded):
    """An N that is not a multiple of 128 masks every B load in the K loop, so
    `mm` widens B's row pitch and the output instead. The padding has to be
    invisible to the caller, and the guards have to keep it out of the shapes
    where the two copies cost more than the masking.
    """
    mm_mod = bound_mm
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.bfloat16, device=flag_gems.device)
    b = torch.randn((K, N), dtype=torch.bfloat16, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a).float(), utils.to_reference(b).float())

    out = flag_gems.mm(a, b)

    assert out.shape == (M, N)
    assert out.is_contiguous(), "a view into the padded buffer reached the caller"
    utils.gems_assert_close(out, ref, torch.bfloat16, reduce_dim=K)

    # A correct result does not say which path produced it, and a guard that
    # silently never fires would pass every assertion above. The key records the
    # N the kernel was launched with, so read it: the key is the eight dtype,
    # device and alignment fields followed by M, N, K and the six strides.
    (key,) = mm_mod._MM_LAUNCH_CACHE
    launched_n = key[9]
    assert launched_n == (
        triton.cdiv(N, mm_mod._PAD_N_TO) * mm_mod._PAD_N_TO if padded else N
    )
