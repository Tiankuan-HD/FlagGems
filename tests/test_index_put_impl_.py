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

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name == "sunrise", reason="Issues #3836: To Fix (Runtime Or LLVM)"
)


if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    ATTN_HEADS = [2]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    ATTN_HEADS = [2, 4, 8, 16, 32]

INDEX_PUT_SHAPE_ACC_FALSE = (
    ((2**28,), ((2**16,),), (2**16,), False),
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

    else:
        for i, shape in enumerate(indices_shape):
            # np.random.choice can accept tuple as size parameter
            index = np.random.choice(
                np.arange(input_shape[i]), size=shape, replace=accumulate
            )
            indices.append(torch.tensor(index, device=flag_gems.device))
        return indices


# Tests for _index_put_impl_
@pytest.mark.index_put_impl_
@pytest.mark.parametrize(
    "input_shape, indices_shape, values_shape, is_bool", INDEX_PUT_SHAPE_ACC_FALSE
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__index_put_impl__acc_false(
    input_shape, indices_shape, values_shape, is_bool, dtype
):
    accumulate = False
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    indices = gen_indices_for_index_put(input_shape, indices_shape, accumulate, is_bool)

    if is_bool:
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
    torch._index_put_impl_(ref_inp, ref_indices, ref_values, accumulate, unsafe=False)
    flag_gems._index_put_impl_(inp, indices, values, accumulate, unsafe=False)

    utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.index_put_impl_
@pytest.mark.parametrize(
    "input_shape, indices_shape, values_shape, is_bool", INDEX_PUT_SHAPE_ACC_TRUE
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test__index_put_impl__acc_true(
    input_shape, indices_shape, values_shape, is_bool, dtype
):
    utils.init_seed(0)

    accumulate = True
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    indices = gen_indices_for_index_put(input_shape, indices_shape, accumulate, is_bool)

    if is_bool:
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
    torch._index_put_impl_(ref_inp, ref_indices, ref_values, accumulate, unsafe=False)
    flag_gems._index_put_impl_(inp, indices, values, accumulate, unsafe=False)

    utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.index_put_impl_
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("unsafe", [True, False])
def test__index_put_impl__unsafe_param(dtype, unsafe):
    """Test _index_put_impl_ with both unsafe=True and unsafe=False"""

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    indices = [torch.randint(0, 32, (8,), device=flag_gems.device)]
    values = torch.randn((8, 64), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(index) for index in indices]
    ref_values = utils.to_reference(values)
    torch._index_put_impl_(
        ref_inp, ref_indices, ref_values, accumulate=False, unsafe=unsafe
    )
    flag_gems._index_put_impl_(inp, indices, values, accumulate=False, unsafe=unsafe)

    utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.index_put_impl_
@pytest.mark.parametrize("dtype", [torch.float32])
def test__index_put_impl__error_all_none(dtype):
    """Test error handling: all None indices for _index_put_impl_"""

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    indices = [None, None]
    values = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)

    with pytest.raises(ValueError):
        flag_gems._index_put_impl_(inp, indices, values, accumulate=False, unsafe=False)


# ── Ascend: launch limits and the semantics the generic kernel drops ───────
#
# _index_put_impl_ is the entry the dispatcher uses for `tensor[idx] = value`,
# and unlike index_put_ the generic version hands the kernel the non-None
# entries of `indices` without first moving the indexed dimensions to the
# front. These are Ascend-only: the generic implementation does not pass them.

_ASCEND_MAX_CORE_DIM = 65535
_ascend_only = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend-specific launch limit"
)


def _core_dim(m, n, block0=2, block1=2048):
    """The flattened grid the Ascend driver sees: coreDim = x * y * z."""
    return -(-m // block0) * -(-n // block1)


def _same_bits(res, ref):
    """Compare the raw bytes; see tests/test_index_put.py for the reasoning."""
    assert res.dtype == ref.dtype, f"{res.dtype} != {ref.dtype}"
    assert res.shape == ref.shape, f"{tuple(res.shape)} != {tuple(ref.shape)}"
    a = res.cpu().reshape(-1).contiguous().view(torch.uint8)
    b = ref.cpu().reshape(-1).contiguous().view(torch.uint8)
    assert torch.equal(a, b), f"{int((a != b).sum())} of {a.numel()} bytes differ"


def _cpu_put(inp, indices, values, accumulate=False):
    """Reference on CPU: torch_npu's own aten::_index_put_impl_ segfaults on
    an index list with a None in the middle, so it cannot referee that case."""
    out = inp.cpu().clone()
    idx = [None if i is None else i.cpu() for i in indices]
    return torch.ops.aten.index_put_(out, idx, values.cpu(), accumulate)


def _dev(t):
    return t.to(flag_gems.device)


@pytest.mark.index_put_impl_
@_ascend_only
def test__index_put_impl__grid_over_core_dim():
    # EE1003: KernelLaunch failed because value 131072 for parameter coreDim
    #         is invalid. Expected value: less than or equal to 65535.
    n, m = 2**20, 2**18
    assert _core_dim(m, 1) > _ASCEND_MAX_CORE_DIM
    inp = torch.zeros(n, dtype=torch.float32, device=flag_gems.device)
    index = _dev(torch.randperm(n)[:m])
    values = torch.arange(1, m + 1, dtype=torch.float32, device=flag_gems.device)

    ref = _cpu_put(inp, [index], values)
    res = flag_gems._index_put_impl_(inp.clone(), [index], values, False, False)
    _same_bits(res, ref)


@pytest.mark.index_put_impl_
@_ascend_only
@pytest.mark.parametrize("accumulate", [False, True])
def test__index_put_impl__empty_index(accumulate):
    # A no-op that the driver still refuses to launch: coreDim must not be 0.
    inp = torch.randn(32, 64, dtype=torch.float32, device=flag_gems.device)
    index = torch.empty(0, dtype=torch.int64, device=flag_gems.device)
    values = torch.empty(0, 64, dtype=torch.float32, device=flag_gems.device)

    res = flag_gems._index_put_impl_(inp.clone(), [index], values, accumulate, False)
    _same_bits(res, inp)


@pytest.mark.index_put_impl_
@_ascend_only
@pytest.mark.parametrize("accumulate", [False, True])
def test__index_put_impl__negative_indices(accumulate):
    inp = torch.zeros(8, 4, dtype=torch.float32, device=flag_gems.device)
    index = torch.tensor([-1, -3, 0, 2], dtype=torch.int64, device=flag_gems.device)
    values = torch.arange(1, 17, dtype=torch.float32, device=flag_gems.device).reshape(
        4, 4
    )

    ref = _cpu_put(inp, [index], values, accumulate)
    res = flag_gems._index_put_impl_(inp.clone(), [index], values, accumulate, False)
    _same_bits(res, ref)


@pytest.mark.index_put_impl_
@_ascend_only
@pytest.mark.parametrize(
    "shape,positions",
    [
        ((32, 64), (1,)),
        ((8, 16, 32), (1,)),
        ((8, 16, 32), (2,)),
        ((8, 16, 32), (0, 2)),
    ],
)
def test__index_put_impl__none_in_indices(shape, positions):
    # The generic version raises here ("the number of sizes provided (1) must
    # be greater or equal to the number of dimensions in the tensor"), because
    # it builds the broadcast target from the index tensors alone.
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
    res = flag_gems._index_put_impl_(inp.clone(), indices, values, False, False)
    _same_bits(res, ref)


@pytest.mark.index_put_impl_
@_ascend_only
@pytest.mark.parametrize("values_kind", ["rows", "all-dims", "scalar"])
def test__index_put_impl__bool_mask(values_kind):
    # A single bool mask over the leading dimension picks whole rows; a mask
    # over every dimension picks scalars and lets values arrive flat, which
    # is the shape plain broadcasting against the selection would reject.
    rows, cols = 64, 8
    inp = torch.zeros(rows, cols, dtype=torch.float32, device=flag_gems.device)

    if values_kind == "rows":
        mask = _dev(torch.arange(rows) % 3 == 0)
        k = int(mask.sum())
        values = _dev(
            torch.arange(1, k * cols + 1, dtype=torch.float32).reshape(k, cols)
        )
    elif values_kind == "all-dims":
        mask = _dev((torch.arange(rows * cols) % 5 == 0).reshape(rows, cols))
        k = int(mask.sum())
        values = _dev(torch.arange(1, k + 1, dtype=torch.float32))
    else:
        mask = _dev(torch.arange(rows) % 3 == 0)
        values = torch.tensor(3.5, dtype=torch.float32, device=flag_gems.device)

    ref = _cpu_put(inp, [mask], values)
    res = flag_gems._index_put_impl_(inp.clone(), [mask], values, False, False)
    _same_bits(res, ref)


@pytest.mark.index_put_impl_
@_ascend_only
def test__index_put_impl__accumulate_duplicates_exact():
    # 85 waves of the 48 vector cores, every slot landing on an exact integer.
    slots, m = 64, 4096
    inp = torch.zeros(slots, dtype=torch.float32, device=flag_gems.device)
    index = _dev(torch.arange(m) % slots)
    values = torch.ones(m, dtype=torch.float32, device=flag_gems.device)

    res = flag_gems._index_put_impl_(inp.clone(), [index], values, True, False)
    _same_bits(res, torch.full((slots,), float(m // slots), dtype=torch.float32))
