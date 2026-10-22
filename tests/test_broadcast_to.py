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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# broadcast_to test cases - define source shapes and target shapes for broadcasting
BROADCAST_TEST_CASES = [
    # (source_shape, target_shape)
    ((1,), (3,)),  # broadcast 1D
    ((3,), (3, 3)),  # broadcast 1D to 2D
    ((1, 3), (2, 3)),  # broadcast 2D
    ((3, 1), (3, 3)),  # broadcast 2D
    ((1, 1), (4, 5)),  # broadcast small to larger
    ((2, 1), (2, 3)),  # broadcast 2D
    ((1, 2, 3), (4, 2, 3)),  # broadcast 3D
    ((4, 1, 1), (4, 5, 6)),  # broadcast 3D
]


@pytest.mark.broadcast_to
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_broadcast_to(dtype):
    # Test with a single broadcast case for each dtype
    src_shape = (1,)
    target_shape = (4,)
    inp = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.broadcast_to(ref_inp, target_shape)
    res_out = flag_gems.broadcast_to(inp, target_shape)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.broadcast_to
@pytest.mark.parametrize("src_shape,target_shape", BROADCAST_TEST_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_broadcast_to_shapes(src_shape, target_shape, dtype):
    inp = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.broadcast_to(ref_inp, target_shape)
    res_out = flag_gems.broadcast_to(inp, target_shape)

    utils.gems_assert_close(res_out, ref_out, dtype)


_ASCEND_MAX_CORE_DIM = 65535
_ascend_only = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend-specific limit"
)


def _assert_same_bits(res, ref):
    """Compare the raw bytes.

    broadcast_to is a pure copy, so the result has to match bit for bit - a
    tolerance would hide -0.0 turning into 0.0, or a NaN losing its payload.
    """
    assert res.dtype == ref.dtype, f"{res.dtype} != {ref.dtype}"
    assert res.shape == ref.shape, f"{tuple(res.shape)} != {tuple(ref.shape)}"
    # reshape(-1) first: view(uint8) refuses a 0-dim tensor, and it also
    # forces a dense copy of a stride-0 view.
    a = res.cpu().reshape(-1).contiguous().view(torch.uint8)
    b = ref.cpu().reshape(-1).contiguous().view(torch.uint8)
    assert torch.equal(a, b), f"{int((a != b).sum())} of {a.numel()} bytes differ"


def _ramp(shape, dtype):
    """Distinct values per element, so a misplaced copy cannot go unnoticed."""
    numel = 1
    for d in shape:
        numel *= d
    if dtype is torch.bool:
        return (torch.arange(numel, device=flag_gems.device) % 2).reshape(shape).bool()
    if dtype.is_complex:
        real = torch.arange(1, numel + 1, device=flag_gems.device, dtype=torch.float32)
        return torch.complex(real, -real).reshape(shape).to(dtype)
    if dtype.is_floating_point:
        base = torch.arange(1, numel + 1, device=flag_gems.device, dtype=torch.float32)
        return (base / 7).reshape(shape).to(dtype)
    high = min(100, torch.iinfo(dtype).max)
    base = torch.arange(numel, device=flag_gems.device, dtype=torch.int64) % high
    return base.reshape(shape).to(dtype)


_INT_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
]


@pytest.mark.broadcast_to
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + _INT_DTYPES + [torch.bool])
def test_broadcast_to_dtypes(dtype):
    inp = _ramp((1, 257), dtype)
    _assert_same_bits(
        flag_gems.broadcast_to(inp, (131, 257)), torch.broadcast_to(inp, (131, 257))
    )


@pytest.mark.broadcast_to
@_ascend_only
@pytest.mark.parametrize(
    "src_shape,target_shape",
    [((1, 257), (131, 257)), ((131, 1), (131, 37)), ((), (5, 7))],
)
def test_broadcast_to_complex(src_shape, target_shape):
    # The generic kernel raises KeyError('complex64') here - triton has no
    # complex pointer type - which a view sidesteps entirely.
    inp = _ramp(src_shape, torch.complex64)
    _assert_same_bits(
        flag_gems.broadcast_to(inp, target_shape), torch.broadcast_to(inp, target_shape)
    )


@pytest.mark.broadcast_to
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_broadcast_to_special_values(dtype):
    # -0.0 and the NaN payload have to survive; an allclose-style check would
    # not notice either.
    inp = torch.tensor(
        [[float("inf"), float("-inf"), float("nan"), -0.0, 0.0, 1.0, -1.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    _assert_same_bits(
        flag_gems.broadcast_to(inp, (37, 7)), torch.broadcast_to(inp, (37, 7))
    )


@pytest.mark.broadcast_to
def test_broadcast_to_subnormal_and_nan_payload():
    raw = torch.tensor(
        [0x7FC00001, 0x7FC00002, 0xFFC00003 - (1 << 32), 0x00000001, 0x007FFFFF],
        dtype=torch.int32,
        device=flag_gems.device,
    )
    inp = raw.view(torch.float32).reshape(1, 5)
    _assert_same_bits(
        flag_gems.broadcast_to(inp, (29, 5)), torch.broadcast_to(inp, (29, 5))
    )


@pytest.mark.broadcast_to
@pytest.mark.parametrize(
    "layout",
    ["transpose", "offset-slice", "strided-slice", "already-expanded", "wide-stride"],
)
def test_broadcast_to_noncontiguous_input(layout):
    base = _ramp((8, 128), torch.float32)
    if layout == "transpose":
        inp, target = _ramp((64, 1), torch.float32).t(), (32, 64)
    elif layout == "offset-slice":
        inp, target = base[3:4], (16, 128)
    elif layout == "strided-slice":
        inp, target = base[2:3, ::2], (16, 64)
    elif layout == "already-expanded":
        inp, target = _ramp((1, 64), torch.float32).expand(8, 64), (8, 64)
    else:
        inp, target = base[1:2, :64], (33, 64)
    _assert_same_bits(
        flag_gems.broadcast_to(inp, target), torch.broadcast_to(inp, target)
    )


@pytest.mark.broadcast_to
@pytest.mark.parametrize(
    "src_shape,target_shape",
    [
        ((3, 1), (3, 4)),  # column broadcast
        ((4,), (2, 3, 4)),  # two leading dims added
        ((), (2, 3)),  # scalar source
        ((1, 5), (4, -1)),  # -1 keeps the existing size
        ((3, 5), (-1, -1)),  # every entry is -1
        ((2, 1, 4), (2, 3, 4)),  # broadcast in the middle
        ((3, 5), (3, 5)),  # no broadcast at all
        ((3, 1), (3, 1)),  # a 1 expanded to 1
        ((1, 2, 1, 2, 1, 2, 1, 2), (2, 2, 2, 2, 2, 2, 2, 2)),  # 8-D, the torch_npu cap
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_broadcast_to_extra_shapes(src_shape, target_shape, dtype):
    inp = _ramp(src_shape, dtype)
    _assert_same_bits(
        flag_gems.broadcast_to(inp, target_shape), torch.broadcast_to(inp, target_shape)
    )


@pytest.mark.broadcast_to
@pytest.mark.parametrize(
    "src_shape,target_shape", [((1, 0), (5, 0)), ((0,), (0,)), ((0,), (3, 0))]
)
def test_broadcast_to_empty(src_shape, target_shape):
    inp = torch.empty(src_shape, dtype=torch.float32, device=flag_gems.device)
    res = flag_gems.broadcast_to(inp, target_shape)
    ref = torch.broadcast_to(inp, target_shape)
    assert res.shape == ref.shape and res.numel() == 0


@pytest.mark.broadcast_to
@pytest.mark.parametrize(
    "src_shape,target_shape",
    [((1, 64), (16, 64)), ((64, 1), (64, 16)), ((), (4, 5)), ((5,), (3, 5))],
)
def test_broadcast_to_matches_aten_view_semantics(src_shape, target_shape):
    # aten::broadcast_to is declared broadcast_to(Tensor(a) self, SymInt[] size)
    # -> Tensor(a): the result aliases its input. Match torch exactly rather
    # than asserting one behaviour or the other, so this test keeps holding if
    # torch ever changes its mind.
    inp = _ramp(src_shape, torch.float32)
    before = inp.clone()
    res = flag_gems.broadcast_to(inp, target_shape)
    ref = torch.broadcast_to(inp, target_shape)
    assert res.shape == ref.shape
    assert res.stride() == ref.stride(), f"{res.stride()} != {ref.stride()}"
    assert (res.data_ptr() == inp.data_ptr()) == (ref.data_ptr() == inp.data_ptr())
    assert res.is_contiguous() == ref.is_contiguous()
    _assert_same_bits(res, ref)
    # Reading the result must not disturb what it was built from.
    _assert_same_bits(inp, before)


@pytest.mark.broadcast_to
@pytest.mark.parametrize(
    "src_shape,target_shape", [((1, 64), (16, 64)), ((64, 1), (64, 16))]
)
def test_broadcast_to_result_feeds_other_ops(src_shape, target_shape):
    # A view carries stride 0 on the broadcast axes, so anything downstream has
    # to read it through its strides rather than as a dense buffer. Push the
    # result through the ops most likely to be handed one.
    inp = _ramp(src_shape, torch.float32)
    res = flag_gems.broadcast_to(inp, target_shape)
    ref = torch.broadcast_to(inp, target_shape)
    other = _ramp(target_shape, torch.float32)
    for name in ("add", "mul", "sub"):
        _assert_same_bits(
            getattr(flag_gems, name)(res, other), getattr(torch, name)(ref, other)
        )
    _assert_same_bits(flag_gems.abs(res), torch.abs(ref))
    _assert_same_bits(res.contiguous(), ref.contiguous())
    torch.testing.assert_close(flag_gems.sum(res), torch.sum(ref))


@pytest.mark.broadcast_to
@pytest.mark.parametrize(
    "src_shape,target_shape",
    [((1, 8), (4, 8)), ((8, 1), (8, 4)), ((3, 5), (3, 5)), ((), (2, 3))],
)
def test_broadcast_to_write_through_matches_aten(src_shape, target_shape):
    """Writing through the result.

    A view shares storage with its input, so a write through the result lands
    in the source; a materialized copy would swallow it. torch does the former.
    Rather than asserting either behaviour, make the same write on both and
    require result and source to come out byte-identical, so the test keeps
    holding whichever way torch resolves the op.
    """
    base = _ramp(src_shape, torch.float32)
    gems_inp, ref_inp = base.clone(), base.clone()
    gems_out = flag_gems.broadcast_to(gems_inp, target_shape)
    ref_out = torch.broadcast_to(ref_inp, target_shape)

    # Fill the first slice. On a broadcast axis several output positions alias
    # one element, but the value written is the same, so the result is still
    # well defined.
    gems_out[0].fill_(-7.0)
    ref_out[0].fill_(-7.0)

    _assert_same_bits(gems_out, ref_out)
    _assert_same_bits(gems_inp, ref_inp)


@pytest.mark.broadcast_to
@pytest.mark.parametrize(
    "src_shape,target_shape", [((1, 8), (4, 8)), ((8, 1), (8, 4))]
)
def test_broadcast_to_tracks_later_input_writes_like_aten(src_shape, target_shape):
    """The other direction: mutate the input after broadcasting.

    A view reflects the change, a copy does not. Compare against torch instead
    of fixing one answer.
    """
    base = _ramp(src_shape, torch.float32)
    gems_inp, ref_inp = base.clone(), base.clone()
    gems_out = flag_gems.broadcast_to(gems_inp, target_shape)
    ref_out = torch.broadcast_to(ref_inp, target_shape)

    gems_inp.fill_(3.5)
    ref_inp.fill_(3.5)

    _assert_same_bits(gems_out, ref_out)


@pytest.mark.broadcast_to
def test_broadcast_to_is_repeatable():
    inp = _ramp((1, 257), torch.float32)
    first = flag_gems.broadcast_to(inp, (131, 257))
    for _ in range(4):
        _assert_same_bits(flag_gems.broadcast_to(inp, (131, 257)), first)


@pytest.mark.broadcast_to
@pytest.mark.parametrize(
    "src_shape,target,exc",
    [
        ((3, 5), (3, 7), RuntimeError),  # non-singleton mismatch
        ((2, 3, 5), (3, 5), RuntimeError),  # target has fewer dims
        ((1, 5), 5, TypeError),  # target is not a sequence
    ],
)
def test_broadcast_to_rejects_bad_sizes(src_shape, target, exc):
    inp = _ramp(src_shape, torch.float32)
    with pytest.raises(exc):
        flag_gems.broadcast_to(inp, target)


# The Ascend driver flattens the launch grid before rtKernelLaunch
# (coreDim = grid.x * grid.y * grid.z) and rejects anything above 65535, which
# a BLOCK_SIZE=1024 copy kernel crosses at 65535 * 1024 + 1 elements. A view
# launches nothing, so the limit cannot be reached - the tests below pin that
# down both structurally and by actually building the oversize outputs.
@pytest.mark.broadcast_to
@_ascend_only
@pytest.mark.parametrize(
    "src_shape,target_shape",
    [
        ((1, 4096), (16383, 4096)),  # 67104768 elements, just under
        ((1, 4096), (16384, 4096)),  # 67108864, the old grid-65536 case
        ((1, 8192), (262145, 8192)),  # 2147491840, past int32 as well
    ],
)
def test_broadcast_to_is_zero_copy(src_shape, target_shape):
    inp = torch.randn(src_shape, dtype=torch.bfloat16, device=flag_gems.device)
    res = flag_gems.broadcast_to(inp, target_shape)
    assert res.data_ptr() == inp.data_ptr(), "broadcast_to materialized a copy"
    assert res.untyped_storage().nbytes() == inp.untyped_storage().nbytes()
    assert 0 in res.stride(), f"no broadcast axis in stride {res.stride()}"


@pytest.mark.broadcast_to
@_ascend_only
@pytest.mark.parametrize(
    "src_shape,target_shape",
    [
        ((1, 4096), (16383, 4096)),  # grid 65532 for the old kernel, just inside
        ((1, 4096), (16384, 4096)),  # grid 65536, which the old kernel failed on
        ((16384, 1), (16384, 4096)),  # same size, column broadcast
        ((1, 2048, 1), (64, 2048, 1024)),  # 3-D, two broadcast axes
    ],
)
def test_broadcast_to_over_core_dim(src_shape, target_shape):
    # expand() is the reference here rather than utils.to_reference: these
    # outputs are 128 MiB and up, and a host round trip would dominate.
    inp = torch.randn(src_shape, dtype=torch.bfloat16, device=flag_gems.device)
    ref_out = inp.expand(target_shape)
    res_out = flag_gems.broadcast_to(inp, target_shape)
    assert res_out.shape == ref_out.shape
    assert torch.equal(res_out, ref_out)
