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

import random
import time

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

INDEX_PUT_SHAPE_ACC_FALSE = (
    ((2**28,), ((2**16,),), (2**16,), False),
    # grid.x = M / BLOCK_SIZE0 must stay within the launch limit, which is
    # 65535 on some backends (e.g. enflame GCU300). M = 2**18 = 262144 exceeds
    # it, so a kernel that chunks grid.x is required here.
    ((2**19,), ((2**18,),), (2**18,), False),
    ((32, 32), ((8,), (8,)), (8,), False),
    ((32, 32), ((8,), (2, 8)), (8,), False),
    ((32, 32), ((2, 8),), (32,), False),
    ((512, 512, 512), ((128,), (128,), (128,)), (128,), False),
    ((512, 512, 512), ((2, 128), (128,), (128,)), (128,), False),
    ((512, 512, 512), ((2, 128),), (512,), False),
    (
        (64, 64, 64),
        (
            (2, 8),
            (2, 8),
        ),
        (2, 8, 64),
        False,
    ),
    ((100,), ((100,),), (100,), True),
    ((32, 32), ((32, 32),), (32, 32), True),
    ((16, 16, 4), ((16, 16, 4),), (16, 16, 4), True),
)

INDEX_PUT_SHAPE_ACC_TRUE = (
    ((2**28,), ((2**16,),), (2**16,), False),
    ((32, 32), ((8,), (8,)), (8,), False),
    ((512, 512, 512), ((128,), (128,), (128,)), (128,), False),
    ((64, 64, 64), ((2, 8), (2, 8), (2, 8)), (2, 8), False),
    ((32, 32), ((32, 32),), (32 * 32,), True),
)

# Make sure every thread has same seed.
random.seed(time.time() // 100)


def gen_indices_for_index_put(input_shape, indices_shape, accumulate, is_bool):
    """
    Generate indices for torch.index_put.
    This function supports multi-dimensional integer index shapes (e.g., (2, 8))
    when is_bool is False, and generates a single boolean mask tensor when
    is_bool is True. This is unlike gen_indices which is designed for
    torch.ops.aten.index that requires broadcastable indices.
    """
    indices = []

    if is_bool:
        mask_shape = indices_shape[0]
        mask = torch.randint(
            0, 2, size=mask_shape, dtype=torch.bool, device=flag_gems.device
        )
        return [mask]

    for i, shape in enumerate(indices_shape):
        # np.random.choice can accept tuple as size parameter
        index = np.random.choice(
            np.arange(input_shape[i]), size=shape, replace=accumulate
        )
        indices.append(torch.tensor(index, device=flag_gems.device))

    return indices


@pytest.mark.index_put
@pytest.mark.parametrize(
    "input_shape, indices_shape, values_shape, is_bool", INDEX_PUT_SHAPE_ACC_FALSE
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_index_put_acc_false(input_shape, indices_shape, values_shape, is_bool, dtype):
    accumulate = False
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    indices = gen_indices_for_index_put(input_shape, indices_shape, accumulate, is_bool)

    if is_bool:
        if flag_gems.vendor_name == "tsingmicro":
            K = indices[0].to(device="cpu").sum().item()
        else:
            K = indices[0].sum().item()

        values = torch.randn(
            (K,), dtype=dtype, device=flag_gems.device, requires_grad=False
        )
    else:
        values = torch.randn(
            values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
        )

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(index) for index in indices]
    ref_values = utils.to_reference(values)
    ref_out = torch.index_put(ref_inp, ref_indices, ref_values, accumulate)
    out = flag_gems.index_put(inp, indices, values, accumulate)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.index_put
@pytest.mark.parametrize(
    "input_shape, indices_shape, values_shape, is_bool", INDEX_PUT_SHAPE_ACC_TRUE
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_put_acc_true(input_shape, indices_shape, values_shape, is_bool, dtype):
    utils.init_seed(0)

    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(24)
        torch.mlu.manual_seed_all(24)

    accumulate = True
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    indices = gen_indices_for_index_put(input_shape, indices_shape, accumulate, is_bool)

    if is_bool:
        if flag_gems.vendor_name == "tsingmicro":
            K = indices[0].to(device="cpu").sum().item()
        else:
            K = indices[0].sum().item()
        values = torch.randn(
            (K,), dtype=dtype, device=flag_gems.device, requires_grad=False
        )
    else:
        values = torch.randn(
            values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
        )

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_indices = [utils.to_reference(index) for index in indices]
    ref_values = utils.to_reference(values, upcast=True)
    ref_out = torch.index_put(ref_inp, ref_indices, ref_values, accumulate)
    out = flag_gems.index_put(inp, indices, values, accumulate)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.index_put_
@pytest.mark.parametrize(
    "input_shape, indices_shape, values_shape, is_bool", INDEX_PUT_SHAPE_ACC_FALSE
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_index_put__acc_false(input_shape, indices_shape, values_shape, is_bool, dtype):
    accumulate = False
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    indices = gen_indices_for_index_put(input_shape, indices_shape, accumulate, is_bool)

    if is_bool:
        if flag_gems.vendor_name == "tsingmicro":
            K = indices[0].to(device="cpu").sum().item()
        else:
            K = indices[0].sum().item()
        values = torch.randn(
            (K,), dtype=dtype, device=flag_gems.device, requires_grad=False
        )
    else:
        values = torch.randn(
            values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
        )

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(index) for index in indices]
    ref_values = utils.to_reference(values)
    torch.index_put_(ref_inp, ref_indices, ref_values, accumulate)
    flag_gems.index_put_(inp, indices, values, accumulate)

    utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.index_put_
@pytest.mark.parametrize(
    "input_shape, indices_shape, values_shape, is_bool", INDEX_PUT_SHAPE_ACC_TRUE
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_put__acc_true(input_shape, indices_shape, values_shape, is_bool, dtype):
    utils.init_seed(0)

    if flag_gems.vendor_name == "metax":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)

    if flag_gems.vendor_name == "mthreads":
        torch.manual_seed(0)
        torch.musa.manual_seed_all(0)

    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)

    accumulate = True
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    indices = gen_indices_for_index_put(input_shape, indices_shape, accumulate, is_bool)

    if is_bool:
        if flag_gems.vendor_name == "tsingmicro":
            K = indices[0].to(device="cpu").sum().item()
        else:
            K = indices[0].sum().item()
        values = torch.randn(
            (K,), dtype=dtype, device=flag_gems.device, requires_grad=False
        )
    else:
        values = torch.randn(
            values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
        )

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_indices = [utils.to_reference(index) for index in indices]
    ref_values = utils.to_reference(values, upcast=True)
    torch.index_put_(ref_inp, ref_indices, ref_values, accumulate)
    flag_gems.index_put_(inp, indices, values, accumulate)

    # BUG #2820: This is a hack
    if flag_gems.vendor_name in ["cambricon", "sunrise"] and dtype == torch.float16:
        inp = utils.to_cpu(inp, ref_inp)
        ref_inp = ref_inp.to(dtype)
        torch.testing.assert_close(inp, ref_inp, atol=3e-3, rtol=3e-2)
    else:
        utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.index_put
@pytest.mark.parametrize("dtype", [torch.float32])
def test_index_put_error_all_none(dtype):
    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    indices = [None, None]
    values = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)

    with pytest.raises(
        ValueError, match="At least one non-None index tensor is required"
    ):
        flag_gems.index_put(inp, indices, values, accumulate=False)


@pytest.mark.index_put_
@pytest.mark.parametrize("dtype", [torch.float32])
def test_index_put__error_all_none(dtype):
    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    indices = [None, None]
    values = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)

    with pytest.raises(
        ValueError, match="At least one non-None index tensor is required"
    ):
        flag_gems.index_put_(inp, indices, values, accumulate=False)


# Format: (input_shape, indices_config)
# 0 in indices_config means a Tensor, 1 in indices_config means None
MIXED_INDEX_SHAPES = [
    ((1024, 1024), (0, 1)),
    ((1024, 1024), (1, 0)),
    ((32, 32, 32), (0, 0, 1)),
    ((32, 32, 32), (0, 1, 0)),
    ((32, 32, 32), (1, 0, 0)),
    ((64, 64, 64), (1, 0, 1)),
    ((12, 12, 12, 12), (1, 0, 0, 0)),
    ((12, 12, 12, 12), (0, 1, 0, 0)),
    ((16, 16, 16, 16), (1, 0, 0, 1)),
    ((16, 16, 16, 16), (0, 1, 1, 0)),
    ((8, 8, 8, 8), (0, 1, 1, 1)),
    ((8, 8, 8, 8), (1, 1, 0, 1)),
]


@pytest.mark.index_put
@pytest.mark.parametrize("input_shape, indices_config", MIXED_INDEX_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_index_put_mixed_none_and_tensor(input_shape, indices_config, dtype):
    accumulate = False
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)

    tensor_dims = [
        input_shape[i] for i, is_none in enumerate(indices_config) if is_none == 0
    ]
    min_dim = min(tensor_dims)
    idx_len = random.randint(3, min(min_dim, 32))
    unique_pool = torch.randperm(min_dim, device=flag_gems.device)[:idx_len]

    indices, ref_indices = [], []
    for i, is_none in enumerate(indices_config):
        if is_none:
            indices.append(None)
            ref_indices.append(slice(None))
        else:
            indices.append(unique_pool)
            ref_indices.append(unique_pool.cpu())

    ref_inp = utils.to_reference(inp)
    target_shape = ref_inp[tuple(ref_indices)].shape

    values = torch.randn(target_shape, dtype=dtype, device=flag_gems.device)
    ref_values = utils.to_reference(values)

    ref_out = ref_inp.clone()
    ref_out[tuple(ref_indices)] = ref_values

    out = flag_gems.index_put(inp, indices, values, accumulate)
    utils.gems_assert_close(out, ref_out, dtype)


# ── Ascend: launch limits and the semantics the generic kernel drops ───────
#
# All of the following are marked Ascend-only. The launch-grid ones describe a
# limit no other backend has, and the rest fail on the generic implementation,
# which these tests are not trying to gate.

_ASCEND_MAX_CORE_DIM = 65535
_ascend_only = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend-specific launch limit"
)


def _core_dim(m, n, block0=2, block1=2048):
    """The flattened grid the Ascend driver sees: coreDim = x * y * z."""
    return -(-m // block0) * -(-n // block1)


def _same_bits(res, ref):
    """Compare the raw bytes.

    With distinct indices index_put is a pure copy, so anything short of an
    exact match is a defect, and a tolerance would also hide -0.0 turning into
    0.0. Duplicate indices are only ever compared on the accumulate path,
    where the expected value is an exact integer.
    """
    assert res.dtype == ref.dtype, f"{res.dtype} != {ref.dtype}"
    assert res.shape == ref.shape, f"{tuple(res.shape)} != {tuple(ref.shape)}"
    # reshape(-1) first: view(uint8) refuses a 0-dim tensor.
    a = res.cpu().reshape(-1).contiguous().view(torch.uint8)
    b = ref.cpu().reshape(-1).contiguous().view(torch.uint8)
    assert torch.equal(a, b), f"{int((a != b).sum())} of {a.numel()} bytes differ"


def _cpu_put(inp, indices, values, accumulate=False):
    """Reference on CPU rather than on the device.

    torch_npu's own aten::_index_put_impl_ segfaults on an index list with a
    None in the middle, so it cannot referee those cases.
    """
    out = inp.cpu().clone()
    idx = [None if i is None else i.cpu() for i in indices]
    return torch.ops.aten.index_put_(out, idx, values.cpu(), accumulate)


def _dev(t):
    return t.to(flag_gems.device)


@pytest.mark.index_put_
@_ascend_only
def test_index_put__grid_over_core_dim():
    # cdiv(M, BLOCK_SIZE0) passes 65535 on its own here, so the generic
    # implementation aborts the process:
    #   EE1003: KernelLaunch failed because value 131072 for parameter coreDim
    #           is invalid. Expected value: less than or equal to 65535.
    n, m = 2**20, 2**18
    assert _core_dim(m, 1) > _ASCEND_MAX_CORE_DIM
    inp = torch.zeros(n, dtype=torch.float32, device=flag_gems.device)
    index = _dev(torch.randperm(n)[:m])
    values = torch.arange(1, m + 1, dtype=torch.float32, device=flag_gems.device)

    ref = _cpu_put(inp, [index], values)
    res = flag_gems.index_put_(inp.clone(), [index], values)
    _same_bits(res, ref)


@pytest.mark.index_put_
@_ascend_only
def test_index_put__grid_over_core_dim_both_axes():
    # Here neither axis is over on its own, their product is: 32768 * 2.
    # values stays a single element and is broadcast, so the case costs one
    # page of memory rather than the 1 GiB a materialized (M, N) would.
    rows, cols, m = 64, 4096, 2**16
    assert _core_dim(m, cols) > _ASCEND_MAX_CORE_DIM
    inp = torch.zeros(rows, cols, dtype=torch.float32, device=flag_gems.device)
    index = _dev(torch.arange(m) % rows)
    values = torch.ones(1, dtype=torch.float32, device=flag_gems.device)

    # Every slot takes m // rows additions of 1.0, an exact integer in fp32.
    ref = torch.full((rows, cols), float(m // rows), dtype=torch.float32)
    res = flag_gems.index_put_(inp.clone(), [index], values, True)
    _same_bits(res, ref)


@pytest.mark.index_put_
@_ascend_only
@pytest.mark.parametrize("accumulate", [False, True])
def test_index_put__empty_index(accumulate):
    # An empty index tensor is a no-op, but it makes the flattened grid 0 and
    # the driver rejects that too: EE1003 ... coreDim ... expected: not equal
    # to 0.
    inp = torch.randn(32, 64, dtype=torch.float32, device=flag_gems.device)
    index = torch.empty(0, dtype=torch.int64, device=flag_gems.device)
    values = torch.empty(0, 64, dtype=torch.float32, device=flag_gems.device)

    res = flag_gems.index_put_(inp.clone(), [index], values, accumulate)
    _same_bits(res, inp)


@pytest.mark.index_put_
@_ascend_only
@pytest.mark.parametrize("accumulate", [False, True])
def test_index_put__negative_indices(accumulate):
    # -1 addresses the last row here as it does everywhere else in torch. The
    # generic kernel masks with `cur_index >= 0` and silently drops the write.
    inp = torch.zeros(8, 4, dtype=torch.float32, device=flag_gems.device)
    index = torch.tensor([-1, -3, 0, 2], dtype=torch.int64, device=flag_gems.device)
    values = torch.arange(1, 17, dtype=torch.float32, device=flag_gems.device).reshape(
        4, 4
    )

    ref = _cpu_put(inp, [index], values, accumulate)
    res = flag_gems.index_put_(inp.clone(), [index], values, accumulate)
    _same_bits(res, ref)


@pytest.mark.index_put_
@_ascend_only
@pytest.mark.parametrize(
    "shape,positions",
    [
        ((32, 64), (1,)),  # trailing index, leading slice
        ((8, 16, 32), (1,)),  # index in the middle
        ((8, 16, 32), (2,)),  # index last
        ((8, 16, 32), (0, 2)),  # indices split by a slice
        ((8, 16, 32), (1, 2)),  # two adjacent indices, not leading
    ],
)
def test_index_put__none_in_indices(shape, positions):
    # A None entry means "take the whole dimension". When the tensor indices
    # are separated by a slice torch puts the broadcast dimensions first,
    # otherwise it leaves them in place; values has to follow either way.
    k = 5
    indices = [None] * len(shape)
    for pos in positions:
        indices[pos] = _dev(torch.randperm(shape[pos])[:k])

    inp = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    key = tuple(slice(None) if i is None else i.cpu() for i in indices)
    probe = inp.cpu()[key]
    values = _dev(
        torch.arange(1, probe.numel() + 1, dtype=torch.float32).reshape(probe.shape)
    )

    ref = _cpu_put(inp, indices, values)
    res = flag_gems.index_put_(inp.clone(), indices, values)
    _same_bits(res, ref)


@pytest.mark.index_put_
@_ascend_only
def test_index_put__accumulate_duplicates_exact():
    # 4096 updates onto 64 slots is 85 waves of the 48 vector cores. Every
    # slot has to land on an exact integer, so a lost atomic add shows up as a
    # differing byte instead of as drift under a tolerance.
    slots, m = 64, 4096
    inp = torch.zeros(slots, dtype=torch.float32, device=flag_gems.device)
    index = _dev(torch.arange(m) % slots)
    values = torch.ones(m, dtype=torch.float32, device=flag_gems.device)

    res = flag_gems.index_put_(inp.clone(), [index], values, True)
    _same_bits(res, torch.full((slots,), float(m // slots), dtype=torch.float32))


@pytest.mark.index_put_
@_ascend_only
@pytest.mark.parametrize("layout", ["transpose", "slice", "strided-slice"])
def test_index_put__noncontiguous_input(layout):
    base = torch.arange(1, 4097, dtype=torch.float32, device=flag_gems.device).reshape(
        64, 64
    )
    if layout == "transpose":
        inp = base.clone().t()
    elif layout == "slice":
        inp = base.clone()[8:24]
    else:
        inp = base.clone()[::2, ::2]

    index = _dev(torch.randperm(inp.shape[0])[:7])
    values = _dev(
        torch.arange(1, 7 * inp.shape[1] + 1, dtype=torch.float32).reshape(
            7, inp.shape[1]
        )
    )

    ref = _cpu_put(inp, [index], values)
    res = flag_gems.index_put_(inp.clone(), [index], values)
    _same_bits(res, ref)


@pytest.mark.index_put_
@_ascend_only
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
        torch.bool,
    ],
)
def test_index_put__dtypes(dtype):
    rows, cols, k = 256, 8, 32
    if dtype is torch.bool:
        inp = torch.zeros(rows, cols, dtype=dtype, device=flag_gems.device)
        values = torch.ones(k, cols, dtype=dtype, device=flag_gems.device)
    elif dtype.is_floating_point:
        inp = torch.zeros(rows, cols, dtype=dtype, device=flag_gems.device)
        values = _dev(
            (torch.arange(1, k * cols + 1, dtype=torch.float32) / 8)
            .reshape(k, cols)
            .to(dtype)
        )
    else:
        inp = torch.zeros(rows, cols, dtype=dtype, device=flag_gems.device)
        values = _dev(torch.arange(1, k * cols + 1, dtype=dtype).reshape(k, cols))

    index = _dev(torch.randperm(rows)[:k])
    ref = _cpu_put(inp, [index], values)
    res = flag_gems.index_put_(inp.clone(), [index], values)
    _same_bits(res, ref)


@pytest.mark.index_put
@_ascend_only
def test_index_put_out_of_place_leaves_input_alone():
    # index_put clones first; the clone is what the kernel writes through.
    inp = torch.zeros(64, 8, dtype=torch.float32, device=flag_gems.device)
    index = _dev(torch.randperm(64)[:9])
    values = _dev(torch.arange(1, 73, dtype=torch.float32).reshape(9, 8))

    ref = _cpu_put(inp, [index], values)
    res = flag_gems.index_put(inp, [index], values)
    _same_bits(res, ref)
    _same_bits(inp, torch.zeros(64, 8, dtype=torch.float32))


@pytest.mark.index_put_
@_ascend_only
@pytest.mark.parametrize("accumulate", [False, True])
def test_index_put__three_dimensional_input(accumulate):
    # Two sliced dimensions behind the index, which the kernel recovers from
    # the lane offset with a chain of % and //.
    inp = torch.zeros(16, 8, 32, dtype=torch.float32, device=flag_gems.device)
    index = _dev(torch.randperm(16)[:5])
    values = _dev(
        torch.arange(1, 5 * 8 * 32 + 1, dtype=torch.float32).reshape(5, 8, 32)
    )

    ref = _cpu_put(inp, [index], values, accumulate)
    res = flag_gems.index_put_(inp.clone(), [index], values, accumulate)
    _same_bits(res, ref)


@pytest.mark.index_put_
@_ascend_only
def test_index_put__permuted_input():
    # The sliced dimensions are not contiguous here, so every lane offset has
    # to go through the strides rather than through a single run.
    base = torch.zeros(4, 8, 16, dtype=torch.float32, device=flag_gems.device)
    inp = base.permute(0, 2, 1)
    assert not inp.is_contiguous()
    index = _dev(torch.randperm(4)[:3])
    values = _dev(
        torch.arange(1, 3 * 16 * 8 + 1, dtype=torch.float32).reshape(3, 16, 8)
    )

    ref = _cpu_put(inp, [index], values)
    res = flag_gems.index_put_(inp.clone(), [index], values)
    _same_bits(res, ref)


@pytest.mark.index_put_
@_ascend_only
def test_index_put__broadcast_scalar_values():
    # values arrives as a stride-0 view of one element; every lane has to read
    # that element rather than walk off it.
    inp = torch.zeros(32, 4, 8, dtype=torch.float32, device=flag_gems.device)
    index = _dev(torch.randperm(32)[:6])
    values = torch.tensor(2.5, dtype=torch.float32, device=flag_gems.device)

    ref = _cpu_put(inp, [index], values)
    res = flag_gems.index_put_(inp.clone(), [index], values)
    _same_bits(res, ref)


@pytest.mark.index_put_
@_ascend_only
@pytest.mark.parametrize("cols", [1, 7, 64, 4096, 5120, 8192, 8193, 70000])
@pytest.mark.parametrize("accumulate", [False, True])
def test_index_put__row_widths(cols, accumulate):
    # Copying whole rows takes a separate kernel that sizes its trailing tile
    # from the row width, and tiles the row once it is wider than one tile.
    # These widths straddle that, and 7 and 8193 are not powers of two.
    rows, k = 8, 5
    inp = torch.zeros(rows, cols, dtype=torch.float32, device=flag_gems.device)
    index = _dev(torch.randperm(rows)[:k])
    # Distinct indices and a value of 1.0: the accumulate result is an exact
    # integer, so a lost or doubled write shows up as a differing byte.
    values = torch.ones(k, cols, dtype=torch.float32, device=flag_gems.device)

    ref = _cpu_put(inp, [index], values, accumulate)
    res = flag_gems.index_put_(inp.clone(), [index], values, accumulate)
    _same_bits(res, ref)


@pytest.mark.index_put_
@_ascend_only
@pytest.mark.parametrize("accumulate", [False, True])
def test_index_put__two_indices_row_copy(accumulate):
    # Two index tensors address a row of the last dimension between them.
    inp = torch.zeros(8, 16, 32, dtype=torch.float32, device=flag_gems.device)
    i0 = _dev(torch.tensor([0, 3, 7]))
    i1 = _dev(torch.tensor([1, 15, 4]))
    values = _dev(torch.arange(1, 3 * 32 + 1, dtype=torch.float32).reshape(3, 32))

    ref = _cpu_put(inp, [i0, i1], values, accumulate)
    res = flag_gems.index_put_(inp.clone(), [i0, i1], values, accumulate)
    _same_bits(res, ref)


@pytest.mark.index_put_
@_ascend_only
def test_index_put__strided_rows_keep_the_shared_kernel():
    # A column slice leaves the row non-contiguous, so the row-copy kernel
    # declines it; the shared kernel still has to get it right.
    base = torch.zeros(32, 64, dtype=torch.float32, device=flag_gems.device)
    inp = base[:, ::2]
    assert inp.stride(-1) == 2
    index = _dev(torch.randperm(32)[:9])
    values = _dev(torch.arange(1, 9 * 32 + 1, dtype=torch.float32).reshape(9, 32))

    ref = _cpu_put(inp, [index], values)
    res = flag_gems.index_put_(inp.clone(), [index], values)
    _same_bits(res, ref)
