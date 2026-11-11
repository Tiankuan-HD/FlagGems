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


@pytest.mark.div_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.true_divide(ref_inp1, ref_inp2)
    res_out = flag_gems.true_divide(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_tensor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_inp1.true_divide_(ref_inp2)
    flag_gems.true_divide_(inp1, inp2)

    utils.gems_assert_close(inp1, ref_inp1, dtype, equal_nan=True)


@pytest.mark.div_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_out(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.empty_like(ref_inp1)
    res_out = torch.empty_like(inp1)

    torch.true_divide(ref_inp1, ref_inp2, out=ref_out)
    flag_gems.true_divide_out(inp1, inp2, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1, False)

    ref_out = torch.true_divide(ref_inp1, inp2)
    res_out = flag_gems.true_divide(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_tensor_scalar_(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1.clone(), False)

    ref_inp1.true_divide_(inp2)
    flag_gems.true_divide_(inp1, inp2)

    utils.gems_assert_close(inp1, ref_inp1, dtype, equal_nan=True)


@pytest.mark.div_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.true_divide(inp1, ref_inp2)
    res_out = flag_gems.true_divide(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.true_divide
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_tensor_dispatch(shape, dtype, caplog):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.ops.aten.true_divide.Tensor(ref_inp1, ref_inp2)
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.true_divide)):
        res_out = flag_gems.true_divide(inp1, inp2)

    assert f"{utils.gems_log_prefix(flag_gems.true_divide)} TRUE_DIVIDE" in caplog.text
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.true_divide_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_tensor_inplace_dispatch(shape, dtype, caplog):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    torch.ops.aten.true_divide_.Tensor(ref_inp1, ref_inp2)
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.true_divide_)):
        res_out = flag_gems.true_divide_(inp1, inp2)

    assert (
        f"{utils.gems_log_prefix(flag_gems.true_divide_)} TRUE_DIVIDE_" in caplog.text
    )
    assert res_out is inp1
    utils.gems_assert_close(inp1, ref_inp1, dtype, equal_nan=True)


@pytest.mark.true_divide
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend UB regression")
@pytest.mark.parametrize("numel", [16384, 1048593])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64]
)
@pytest.mark.parametrize(
    "operands", ["tensor_tensor", "tensor_scalar", "scalar_tensor"]
)
def test_true_divide_ascend_ub(numel, dtype, operands):
    # Exercise the old overflowing tile and multiple persistent iterations with
    # a masked tail. Integer inputs also need UB space for conversion to FP32.
    a = torch.randint(-100, 100, (numel,), dtype=torch.int64).to(dtype)
    b = torch.randint(1, 100, (numel,), dtype=torch.int64).to(dtype)
    if operands == "tensor_scalar":
        b = 2.5
    elif operands == "scalar_tensor":
        a = 2.5
    expected = torch.true_divide(a, b)
    a_device = a.to(flag_gems.device) if isinstance(a, torch.Tensor) else a
    b_device = b.to(flag_gems.device) if isinstance(b, torch.Tensor) else b

    result = flag_gems.true_divide(a_device, b_device)

    torch.testing.assert_close(result.cpu(), expected, equal_nan=True)
