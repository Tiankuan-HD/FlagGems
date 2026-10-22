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

"""Ascend-specialized ``index_put`` / ``index_put_`` / ``_index_put_impl_``.

The generic implementation launches ``grid = (cdiv(M, BLOCK_SIZE0),
cdiv(N, BLOCK_SIZE1))`` with no upper bound, where ``M`` is the number of
indices and ``N`` the volume of the trailing sliced dimensions. The Ascend
driver flattens the grid into ``coreDim = grid.x * grid.y * grid.z`` before
``rtKernelLaunch`` and accepts only ``1 <= coreDim <= 65535``, so the generic
path aborts the process at both ends: ``coreDim > 65535`` once
``cdiv(M, 2) * cdiv(N, 2048)`` passes the limit, and ``coreDim == 0`` for an
empty index tensor, which is a legal no-op elsewhere.

This variant caps the launch at the largest grid the driver accepts and
serializes the remaining chunks in inner loops, the same shape as
``unsafe_index_put``. Capping at the driver's own limit rather than at
something smaller is what keeps the result identical to the generic
implementation wherever the generic implementation can run at all: the grid,
and with it the order the atomics land in on the accumulate path, is the same
one it would have launched. Only grids it could not have launched are
reshaped. Sweeping the cap from one wave (48 programs here) up to 65535 moves
the 1-D M=65536 case by 1%, so nothing is paid for the larger grid.

A third kernel takes the opposite shape, where every dimension is indexed
and each lane writes a single element. The store stays scattered, but the
index and value reads are contiguous and only vectorize once the offsets are
plain 1-D vectors; the shared kernel wraps them in a ``(BLOCK_SIZE0, 1)``
tile whose second axis never spans anything, which at ``BLOCK_SIZE0 = 2``
leaves two elements per program. For 65536 indices that is 2385 us against
352, which is what ``aclnnIndexPutImpl`` takes. It is used for plain stores
only: the order of stores is not observable, the order of atomic adds is.

Where the put reduces to copying whole contiguous rows -- one trailing
dimension, rank-1 indices, both operands innermost-contiguous -- a second
kernel takes over, and it is worth about two orders of magnitude. The
difference is that it addresses a row from a *scalar* base: one program step
per index, so ``cur_index`` is a scalar and the lane offset is a plain
``arange``, which the backend turns into contiguous vector moves. The shared
kernel builds a two-dimensional table of addresses instead, each lane
carrying its own row base through a ``%``/``//`` chain, and the backend falls
back to scalar moves -- the "discrete access mapped to scalar" case the
Triton-Ascend migration guide warns about. On ``(1024, 5120)`` bf16 that is
17.2 ms against 90 us, with ``aclnnIndexPutImpl`` at 55 us; both halves are
needed, a scalar base while the ``%`` stays costs 15.1 ms and dropping the
``%`` under a vector base costs 15.8 ms.

Two correctness gaps of the generic implementation are fixed here as well:

* negative indices are part of the aten semantics, but the generic kernel
  masks them away with ``cur_index >= 0`` and drops those writes silently.
  The kernel below wraps them with ``cur_index + input_shape`` first.
* ``_index_put_impl_`` passes the non-``None`` entries of ``indices`` straight
  to the kernel, without the transposition that ``index_put_`` performs, while
  the kernel assumes the tensor indices occupy the leading dimensions. Here
  all three entry points share one preprocessing step, so an index list with
  ``None`` in it behaves the same whichever is called.
"""

import importlib
import logging
import math
from typing import Any, List, Optional, Sequence, Tuple

import torch

from flag_gems.ops.index_put import generate_imports
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)

# The flattened grid the driver will accept, i.e. the hard ceiling the capped
# launch has to stay under even when the device reports more cores than that.
MAX_CORE_DIM = 65535

# Trailing tile of the row-wise kernel. One tile per row is what makes it
# fast: against a 5120-wide row, 2048 costs 3x the time 8192 does. 32768 does
# not get through TritonToLinalg at all.
FAST_TRAILING_BLOCK = 8192

# Index tile of the scatter kernel. 512 lands on the vendor operator's time
# for 65536 indices; 64 and 2048 are both about 10% worse, and 8192 does not
# compile.
SCATTER_BLOCK = 512


def generate_index_put_kernel_ascend(
    inp_rank, indices_len, index_rank, kernel_name: str, code: IndentedBuffer
):
    code.writeline("@libentry()")
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        args = ["input_ptr,"]
        args += [f"indices{i}_ptr," for i in range(indices_len)]
        args += ["values_ptr,"]
        args += [f"input_shape{i}," for i in range(inp_rank)]
        for i in range(indices_len):
            args += [f"indices{i}_shape{j}," for j in range(index_rank)]
        args += [f"input_stride{i}," for i in range(inp_rank)]
        for i in range(indices_len):
            args += [f"indices{i}_stride{j}," for j in range(index_rank)]
        args += [
            f"values_stride{i}," for i in range(index_rank + inp_rank - indices_len)
        ]
        args += [
            "M,",
            "N,",
            "IS_ACCUMULATE: tl.constexpr,",
            "BLOCK_SIZE0: tl.constexpr = 2,",
            "BLOCK_SIZE1: tl.constexpr = 2048,",
        ]
        code.writelines(args)
    code.writeline("):")

    with code.indent():
        # ext.program_id is int64: a capped grid keeps the program count small,
        # but the offsets it builds still run over the whole tensor.
        code.writeline("pid0 = ext.program_id(axis=0)")
        code.writeline("pid1 = ext.program_id(axis=1)")
        code.writeline("total0 = tl.cdiv(M, BLOCK_SIZE0)")
        code.writeline("iter0 = tl.cdiv(total0, ext.num_programs(axis=0).to(tl.int32))")
        code.writeline("for i0 in range(0, iter0):")
        with code.indent():
            code.writeline(
                "offset0 = (pid0 * iter0 + i0) * BLOCK_SIZE0"
                " + tl.arange(0, BLOCK_SIZE0)[:, None]"
            )
            if inp_rank == indices_len:
                code.writeline("offset1 = pid1 * 1 + tl.arange(0, 1)[None, :]")
                _emit_index_put_body(code, inp_rank, indices_len, index_rank)
            else:
                code.writeline("total1 = tl.cdiv(N, BLOCK_SIZE1)")
                code.writeline(
                    "iter1 = tl.cdiv(total1, ext.num_programs(axis=1).to(tl.int32))"
                )
                code.writeline("for i1 in range(0, iter1):")
                with code.indent():
                    code.writeline(
                        "offset1 = (pid1 * iter1 + i1) * BLOCK_SIZE1"
                        " + tl.arange(0, BLOCK_SIZE1)[None, :]"
                    )
                    _emit_index_put_body(code, inp_rank, indices_len, index_rank)

    code.newline()
    code.newline()
    return code


def _emit_index_put_body(
    code: IndentedBuffer, inp_rank: int, indices_len: int, index_rank: int
):
    """Emit index decomposition, loads and the atomic/store at current indent."""
    code.writeline("cur_idx = offset0")
    for i in range(index_rank - 1, -1, -1):
        code.writeline(f"indices_idx{i} = cur_idx % indices0_shape{i}")
        code.writeline(f"cur_idx = cur_idx // indices0_shape{i}")
    code.newline()
    code.writeline("cur_idx = offset1")
    for i in range(inp_rank - 1, indices_len - 1, -1):
        code.writeline(f"input_idx{i} = cur_idx % input_shape{i}")
        code.writeline(f"cur_idx = cur_idx // input_shape{i}")
    code.newline()
    code.writeline("mask0 = offset0 < M")
    for i in range(indices_len):
        comp = [f"indices_idx{j} * indices{i}_stride{j}" for j in range(index_rank)]
        code.writeline(
            f"cur_index{i} = tl.load(indices{i}_ptr + {' + '.join(comp)}, mask=mask0, other=0)"
        )
    code.newline()
    # Negative indices count from the end, as everywhere else in torch. The
    # generic kernel has no wrap and its `>= 0` mask drops them instead.
    for i in range(indices_len):
        code.writeline(
            f"cur_index{i} = tl.where(cur_index{i} < 0,"
            f" cur_index{i} + input_shape{i}, cur_index{i})"
        )
    code.newline()
    index_mask = [
        f"(cur_index{i} >= 0) & (cur_index{i} < input_shape{i})"
        for i in range(indices_len)
    ]
    code.writeline(f"index_mask = {' & '.join(index_mask)}")
    code.writeline("mask1 = offset1 < N")
    code.writeline("mask = index_mask & mask0 & mask1")
    code.newline()
    comp = [f"cur_index{i} * input_stride{i}" for i in range(indices_len)]
    comp += [f"input_idx{i} * input_stride{i}" for i in range(indices_len, inp_rank)]
    code.writeline(f"input_offset = {' + '.join(comp)}")
    comp = [f"indices_idx{i} * values_stride{i}" for i in range(index_rank)]
    comp += [
        f"input_idx{indices_len + i} * values_stride{index_rank + i}"
        for i in range(inp_rank - indices_len)
    ]
    code.writeline(f"values_offset = {' + '.join(comp)}")
    code.newline()
    code.writeline("cur_value = tl.load(values_ptr + values_offset, mask=mask)")
    code.writeline("if IS_ACCUMULATE:")
    with code.indent():
        code.writeline("tl.atomic_add(input_ptr + input_offset, cur_value, mask=mask)")
    code.writeline("else:")
    with code.indent():
        code.writeline("tl.store(input_ptr + input_offset, cur_value, mask=mask)")


def generate_index_put_wrapper_ascend(
    inp_rank,
    indices_len,
    index_rank,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
):
    code.writeline(
        f"def {wrapper_name}(input, indices, values, accumulate, max_programs):"
    )
    with code.indent():
        code.writeline("input_shape = input.shape")
        code.writeline("input_stride = input.stride()")
        for i in range(indices_len):
            code.writeline(f"indices{i}_shape = indices[{i}].shape")
            code.writeline(f"indices{i}_stride = indices[{i}].stride()")
        code.writeline("values_shape = values.shape")
        code.writeline("values_stride = values.stride()")
        code.writeline("M = indices[0].numel()")
        code.writeline(f"N = volume(input_shape[{indices_len}: ])")
        code.newline()
        # Cap the flattened grid at max_programs; the kernel serializes the
        # remaining chunks in its inner loops.
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline("min(triton.cdiv(M, meta['BLOCK_SIZE0']), max_programs),")
            code.writeline("min(")
            with code.indent():
                code.writeline("triton.cdiv(N, meta['BLOCK_SIZE1']),")
                code.writeline(
                    "max(1, max_programs // min("
                    "triton.cdiv(M, meta['BLOCK_SIZE0']), max_programs)"
                    "),"
                )
            code.writeline("),")
        code.writeline(")")
        code.newline()
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            args = ["input,"]
            args += [f"indices[{i}]," for i in range(indices_len)]
            args += ["values,"]
            args += [f"input_shape[{i}]," for i in range(inp_rank)]
            for i in range(indices_len):
                args += [f"indices{i}_shape[{j}]," for j in range(index_rank)]
            args += [f"input_stride[{i}]," for i in range(inp_rank)]
            for i in range(indices_len):
                args += [f"indices{i}_stride[{j}]," for j in range(index_rank)]
            args += [
                f"values_stride[{i}],"
                for i in range(index_rank + inp_rank - indices_len)
            ]
            args += ["M,", "N,", "accumulate==True,"]
            code.writelines(args)
        code.writeline(")")
        code.writeline("return input")
    code.newline()
    code.newline()
    return code


def generate_index_put_scatter_kernel(
    indices_len, index_rank, kernel_name: str, code: IndentedBuffer
):
    """Every dimension is indexed, so each lane writes one element.

    The store is scattered and stays that way, but the index and value reads
    are contiguous, and they only vectorize once the offsets are plain 1-D
    vectors: the shared kernel wraps them in a ``(BLOCK_SIZE0, 1)`` tile whose
    second axis never spans anything, and at ``BLOCK_SIZE0 = 2`` that leaves
    two elements per program. For 65536 indices this is 2385 us against 352.
    """
    code.writeline("@libentry()")
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        args = ["input_ptr,"]
        args += [f"indices{i}_ptr," for i in range(indices_len)]
        args += ["values_ptr,"]
        args += [f"input_shape{i}," for i in range(indices_len)]
        for i in range(indices_len):
            args += [f"indices{i}_shape{j}," for j in range(index_rank)]
        args += [f"input_stride{i}," for i in range(indices_len)]
        for i in range(indices_len):
            args += [f"indices{i}_stride{j}," for j in range(index_rank)]
        args += [f"values_stride{j}," for j in range(index_rank)]
        args += ["M,", "BLOCK_SIZE0: tl.constexpr,"]
        code.writelines(args)
    code.writeline("):")

    with code.indent():
        code.writeline("pid0 = ext.program_id(axis=0)")
        code.writeline("total0 = tl.cdiv(M, BLOCK_SIZE0)")
        code.writeline("iter0 = tl.cdiv(total0, ext.num_programs(axis=0).to(tl.int32))")
        code.writeline("for i0 in range(0, iter0):")
        with code.indent():
            code.writeline(
                "offset0 = (pid0 * iter0 + i0) * BLOCK_SIZE0"
                " + tl.arange(0, BLOCK_SIZE0)"
            )
            code.writeline("mask0 = offset0 < M")
            code.writeline("cur_idx = offset0")
            for j in range(index_rank - 1, -1, -1):
                code.writeline(f"indices_idx{j} = cur_idx % indices0_shape{j}")
                code.writeline(f"cur_idx = cur_idx // indices0_shape{j}")
            code.newline()
            for i in range(indices_len):
                comp = " + ".join(
                    f"indices_idx{j} * indices{i}_stride{j}" for j in range(index_rank)
                )
                code.writeline(
                    f"cur_index{i} = tl.load(indices{i}_ptr + {comp},"
                    " mask=mask0, other=0)"
                )
                code.writeline(
                    f"cur_index{i} = tl.where(cur_index{i} < 0,"
                    f" cur_index{i} + input_shape{i}, cur_index{i})"
                )
            index_mask = " & ".join(
                f"(cur_index{i} >= 0) & (cur_index{i} < input_shape{i})"
                for i in range(indices_len)
            )
            code.writeline(f"mask = mask0 & {index_mask}")
            code.newline()
            code.writeline(
                "input_offset = "
                + " + ".join(
                    f"cur_index{i} * input_stride{i}" for i in range(indices_len)
                )
            )
            code.writeline(
                "values_offset = "
                + " + ".join(
                    f"indices_idx{j} * values_stride{j}" for j in range(index_rank)
                )
            )
            code.writeline("cur_value = tl.load(values_ptr + values_offset, mask=mask)")
            code.writeline("tl.store(input_ptr + input_offset, cur_value, mask=mask)")
    code.newline()
    code.newline()
    return code


def generate_index_put_scatter_wrapper(
    indices_len, index_rank, wrapper_name: str, kernel_name: str, code: IndentedBuffer
):
    code.writeline(
        f"def {wrapper_name}(input, indices, values, accumulate, max_programs):"
    )
    with code.indent():
        code.writeline("input_shape = input.shape")
        code.writeline("input_stride = input.stride()")
        for i in range(indices_len):
            code.writeline(f"indices{i}_shape = indices[{i}].shape")
            code.writeline(f"indices{i}_stride = indices[{i}].stride()")
        code.writeline("values_stride = values.stride()")
        code.writeline("M = indices[0].numel()")
        code.newline()
        code.writeline(f"block0 = min(triton.next_power_of_2(M), {SCATTER_BLOCK})")
        code.writeline("grid = (min(triton.cdiv(M, block0), max_programs),)")
        code.newline()
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            args = ["input,"]
            args += [f"indices[{i}]," for i in range(indices_len)]
            args += ["values,"]
            args += [f"input_shape[{i}]," for i in range(indices_len)]
            for i in range(indices_len):
                args += [f"indices{i}_shape[{j}]," for j in range(index_rank)]
            args += [f"input_stride[{i}]," for i in range(indices_len)]
            for i in range(indices_len):
                args += [f"indices{i}_stride[{j}]," for j in range(index_rank)]
            args += [f"values_stride[{j}]," for j in range(index_rank)]
            args += ["M,"]
            code.writelines(args)
            code.writeline("BLOCK_SIZE0=block0,")
        code.writeline(")")
        code.writeline("return input")
    code.newline()
    code.newline()
    return code


def generate_scatter_code_ascend(
    indices_len: int,
    index_rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
):
    code = generate_imports(code)
    generate_index_put_scatter_kernel(indices_len, index_rank, kernel_name, code)
    generate_index_put_scatter_wrapper(
        indices_len, index_rank, wrapper_name, kernel_name, code
    )
    return code


def generate_index_put_rowwise_kernel(
    indices_len, kernel_name: str, code: IndentedBuffer
):
    """One index per loop step, so the row base is a scalar.

    Only emitted for ``inp_rank == indices_len + 1`` with rank-1 indices: the
    whole put is then "copy row r of values onto row cur_index of input".
    """
    inp_rank = indices_len + 1
    code.writeline("@libentry()")
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        args = ["input_ptr,"]
        args += [f"indices{i}_ptr," for i in range(indices_len)]
        args += ["values_ptr,"]
        args += [f"input_shape{i}," for i in range(inp_rank)]
        args += [f"indices{i}_shape0," for i in range(indices_len)]
        args += [f"input_stride{i}," for i in range(inp_rank)]
        args += [f"indices{i}_stride0," for i in range(indices_len)]
        args += ["values_stride0,", "values_stride1,"]
        args += [
            "M,",
            "N,",
            "IS_ACCUMULATE: tl.constexpr,",
            "BLOCK_SIZE1: tl.constexpr,",
        ]
        code.writelines(args)
    code.writeline("):")

    with code.indent():
        code.writeline("pid0 = ext.program_id(axis=0)")
        code.writeline("pid1 = ext.program_id(axis=1)")
        code.writeline("iter0 = tl.cdiv(M, ext.num_programs(axis=0).to(tl.int32))")
        code.writeline("for i0 in range(0, iter0):")
        with code.indent():
            code.writeline("r = pid0 * iter0 + i0")
            code.writeline("if r < M:")
            with code.indent():
                for i in range(indices_len):
                    code.writeline(
                        f"cur_index{i} = tl.load("
                        f"indices{i}_ptr + r * indices{i}_stride0)"
                    )
                    code.writeline(
                        f"cur_index{i} = tl.where(cur_index{i} < 0,"
                        f" cur_index{i} + input_shape{i}, cur_index{i})"
                    )
                guard = " & ".join(
                    f"(cur_index{i} >= 0) & (cur_index{i} < input_shape{i})"
                    for i in range(indices_len)
                )
                code.writeline(f"if {guard}:")
                with code.indent():
                    base = " + ".join(
                        f"cur_index{i} * input_stride{i}" for i in range(indices_len)
                    )
                    code.writeline(f"row_offset = {base}")
                    code.writeline("total1 = tl.cdiv(N, BLOCK_SIZE1)")
                    code.writeline(
                        "iter1 = tl.cdiv(total1,"
                        " ext.num_programs(axis=1).to(tl.int32))"
                    )
                    code.writeline("for i1 in range(0, iter1):")
                    with code.indent():
                        code.writeline(
                            "offset1 = (pid1 * iter1 + i1) * BLOCK_SIZE1"
                            " + tl.arange(0, BLOCK_SIZE1)"
                        )
                        code.writeline("mask1 = offset1 < N")
                        code.writeline(
                            "input_offset = row_offset"
                            f" + offset1 * input_stride{indices_len}"
                        )
                        code.writeline(
                            "values_offset = r * values_stride0"
                            " + offset1 * values_stride1"
                        )
                        code.writeline(
                            "cur_value = tl.load(values_ptr + values_offset, mask=mask1)"
                        )
                        code.writeline("if IS_ACCUMULATE:")
                        with code.indent():
                            code.writeline(
                                "tl.atomic_add(input_ptr + input_offset,"
                                " cur_value, mask=mask1)"
                            )
                        code.writeline("else:")
                        with code.indent():
                            code.writeline(
                                "tl.store(input_ptr + input_offset,"
                                " cur_value, mask=mask1)"
                            )
    code.newline()
    code.newline()
    return code


def generate_index_put_rowwise_wrapper(
    indices_len, wrapper_name: str, kernel_name: str, code: IndentedBuffer
):
    inp_rank = indices_len + 1
    code.writeline(
        f"def {wrapper_name}(input, indices, values, accumulate, max_programs):"
    )
    with code.indent():
        code.writeline("input_shape = input.shape")
        code.writeline("input_stride = input.stride()")
        for i in range(indices_len):
            code.writeline(f"indices{i}_stride = indices[{i}].stride()")
        code.writeline("values_stride = values.stride()")
        code.writeline("M = indices[0].numel()")
        code.writeline(f"N = input_shape[{indices_len}]")
        code.newline()
        # One tile per row wherever it fits: a trailing tile of 2048 against a
        # 5120-wide row costs 3x the time of one that covers the row in one go,
        # and 32768 does not survive TritonToLinalg.
        code.writeline(
            "block1 = min(triton.next_power_of_2(N), " f"{FAST_TRAILING_BLOCK})"
        )
        code.writeline("g0 = min(M, max_programs)")
        code.writeline("grid = (")
        with code.indent():
            code.writeline("g0,")
            code.writeline("min(triton.cdiv(N, block1), max(1, max_programs // g0)),")
        code.writeline(")")
        code.newline()
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            args = ["input,"]
            args += [f"indices[{i}]," for i in range(indices_len)]
            args += ["values,"]
            args += [f"input_shape[{i}]," for i in range(inp_rank)]
            args += [f"indices[{i}].shape[0]," for i in range(indices_len)]
            args += [f"input_stride[{i}]," for i in range(inp_rank)]
            args += [f"indices{i}_stride[0]," for i in range(indices_len)]
            args += ["values_stride[0],", "values_stride[1],"]
            args += ["M,", "N,", "accumulate==True,"]
            code.writelines(args)
            code.writeline("BLOCK_SIZE1=block1,")
        code.writeline(")")
        code.writeline("return input")
    code.newline()
    code.newline()
    return code


def generate_rowwise_code_ascend(
    indices_len: int, wrapper_name: str, kernel_name: str, code: IndentedBuffer
):
    code = generate_imports(code)
    generate_index_put_rowwise_kernel(indices_len, kernel_name, code)
    generate_index_put_rowwise_wrapper(indices_len, wrapper_name, kernel_name, code)
    return code


def generate_code_ascend(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
):
    inp_rank = inputs[0].ndim
    tensor_indices = [idx for idx in inputs[1] if idx is not None]
    indices_len = len(tensor_indices)
    if indices_len == 0:
        raise ValueError("At least one non-None index tensor is required")
    index_rank = tensor_indices[0].ndim
    code = generate_imports(code)
    generate_index_put_kernel_ascend(
        inp_rank, indices_len, index_rank, kernel_name, code
    )
    generate_index_put_wrapper_ascend(
        inp_rank, indices_len, index_rank, wrapper_name, kernel_name, code
    )
    return code


class AscendIndexPutFunction:
    def __init__(self):
        self.overloads = {}

    def arg_key(self, inp, tensor_indices, variant):
        indices_len = len(tensor_indices)
        index_rank = 0 if indices_len == 0 else tensor_indices[0].ndim
        key = f"inp_rank_{inp.ndim}_indices_len_{indices_len}_index_rank_{index_rank}"
        return f"{key}_{variant}" if variant else key

    def __call__(self, inp, tensor_indices, values, accumulate, variant=""):
        key = self.arg_key(inp, tensor_indices, variant)
        overload = self.overloads.get(key)
        if overload is None:
            code = IndentedBuffer()
            if variant == "rowwise":
                code = generate_rowwise_code_ascend(
                    len(tensor_indices),
                    "_index_put_wrapper",
                    "_index_put_rowwise_jit_function",
                    code,
                )
            elif variant == "scatter":
                code = generate_scatter_code_ascend(
                    len(tensor_indices),
                    tensor_indices[0].ndim,
                    "_index_put_wrapper",
                    "_index_put_scatter_jit_function",
                    code,
                )
            else:
                code = generate_code_ascend(
                    (inp, tensor_indices, values),
                    "_index_put_wrapper",
                    "_index_put_jit_function",
                    code,
                )
            file_name = f"index_put_ascend_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_index_put_ascend_{key}", file_path
            )
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_put_wrapper")
            self.overloads[key] = overload

        return overload(inp, tensor_indices, values, accumulate, MAX_CORE_DIM)


_ascend_index_put_func = AscendIndexPutFunction()


def _expand_masks(
    inp: torch.Tensor, indices: Sequence[Optional[torch.Tensor]]
) -> List[Optional[torch.Tensor]]:
    """Move indices onto the input device and turn bool/int8 masks into
    explicit integer indices, one per masked dimension."""
    expanded: List[Optional[torch.Tensor]] = []
    for index in indices:
        if index is None:
            expanded.append(None)
        elif not torch.is_tensor(index):
            raise TypeError(
                "tensors used as indices must be long, int, byte or bool tensors"
            )
        else:
            if index.device != inp.device:
                index = index.to(inp.device)
            if index.dtype in (torch.bool, torch.int8):
                expanded.extend(index.nonzero(as_tuple=True))
            else:
                expanded.append(index)
    return expanded


def _preprocess(inp, indices, values):
    """Bring ``(inp, indices, values)`` into the layout the kernel assumes:
    the tensor indices lead, the sliced dimensions trail, and ``values`` is
    shaped ``broadcast_shape + slice_shape``.

    Returns ``(inp_view, tensors, values, slice_shape)``.
    """
    indices = _expand_masks(inp, list(indices))

    if not indices:
        raise ValueError("At least one index tensor is required")
    if len(indices) > inp.ndim:
        raise IndexError("too many indices for tensor of dimension {}".format(inp.ndim))
    if len(indices) < inp.ndim:
        indices.extend([None] * (inp.ndim - len(indices)))

    tensor_pos = [i for i, x in enumerate(indices) if x is not None]
    if not tensor_pos:
        raise ValueError("At least one non-None index tensor is required")

    tensor_indices = [indices[i] for i in tensor_pos]
    if len(tensor_indices) > 1:
        broadcasted = torch.broadcast_tensors(*tensor_indices)
        for i, pos in enumerate(tensor_pos):
            indices[pos] = broadcasted[i]

    # The kernel decomposes offsets assuming the indexed dimensions come first,
    # so anything else has to be permuted into that layout first.
    is_contiguous = (tensor_pos[-1] - tensor_pos[0] + 1) == len(tensor_pos)
    need_transpose = not is_contiguous or indices[0] is None

    if need_transpose:
        perm_order = tensor_pos + [i for i, x in enumerate(indices) if x is None]
        inp_view = inp.permute(perm_order)
        final_indices = [indices[i] for i in tensor_pos] + [None] * (
            len(indices) - len(tensor_pos)
        )
    else:
        inp_view = inp
        final_indices = indices

    tensors = [x for x in final_indices if x is not None]
    broadcast_shape = list(tensors[0].shape)
    slice_shape = [inp_view.shape[i] for i, x in enumerate(final_indices) if x is None]

    if values.device != inp.device:
        values = values.to(inp.device)

    if need_transpose and is_contiguous:
        # torch broadcasts values against the *unpermuted* layout, so build
        # that shape first and apply the same permutation to it.
        num_before = tensor_pos[0]
        before_dims = slice_shape[:num_before]
        after_dims = slice_shape[num_before:]
        values = values.broadcast_to(before_dims + broadcast_shape + after_dims)
        num_b, num_t = len(before_dims), len(broadcast_shape)
        values = values.permute(
            list(range(num_b, num_b + num_t))
            + list(range(0, num_b))
            + list(range(num_b + num_t, values.ndim))
        )
    else:
        values = values.broadcast_to(broadcast_shape + slice_shape)

    return inp_view, tensors, values, slice_shape


def _fold_stride(tensor, start):
    """The stride a fold of ``tensor``'s dimensions from ``start`` on would
    have, or ``None`` when they cannot be folded without copying.

    Dimensions fold when they form one uniformly strided run (the usual
    contiguous case, and a strided slice of one), or when every one of them is
    a stride-0 broadcast, which folds to stride 0.
    """
    shape, stride, ndim = tensor.shape, tensor.stride(), tensor.ndim
    if ndim - start <= 1:
        return None
    if all(stride[d] == 0 for d in range(start, ndim)):
        return 0
    # A size-1 dimension carries an arbitrary stride, so start from the
    # innermost dimension that actually spans something.
    inner = next((d for d in range(ndim - 1, start - 1, -1) if shape[d] != 1), None)
    if inner is None:
        return 1
    expected = stride[inner]
    for d in range(inner, start - 1, -1):
        if shape[d] == 1:
            continue
        if stride[d] != expected:
            return None
        expected *= shape[d]
    return stride[inner]


def _fold(tensor, start, folded_size, folded_stride):
    return tensor.as_strided(
        list(tensor.shape[:start]) + [folded_size],
        list(tensor.stride()[:start]) + [folded_stride],
        tensor.storage_offset(),
    )


def _fold_trailing(inp_view, values, slice_shape):
    """Collapse the sliced dimensions into one, as views.

    Only applied when the result is something the row-wise kernel can take --
    folding also reshapes the launch of the shared kernel, and that would move
    the order its atomics land in for cases the row-wise kernel then declines.
    The input is written through, so a copy would land in the wrong place;
    both operands have to fold as views or the fold is dropped.
    """
    if len(slice_shape) <= 1:
        return inp_view, values, slice_shape
    lead = inp_view.ndim - len(slice_shape)
    inp_stride = _fold_stride(inp_view, lead)
    val_stride = _fold_stride(values, values.ndim - len(slice_shape))
    if inp_stride != 1 or val_stride != 1:
        return inp_view, values, slice_shape
    folded = math.prod(slice_shape)
    return (
        _fold(inp_view, lead, folded, inp_stride),
        _fold(values, values.ndim - len(slice_shape), folded, val_stride),
        [folded],
    )


def _rowwise_eligible(inp_view, tensors, values, slice_shape):
    """Whether the put is a row copy the scalar-base kernel can take.

    It needs exactly one sliced dimension and rank-1 indices, so a row is one
    contiguous run addressed from a scalar base, and it needs that run to be
    contiguous on both sides. A ``values`` broadcast along the row has
    ``stride(-1) == 0``, which collapses the load to a single address and does
    not survive TritonToLinalg, so those keep the shared kernel.
    """
    return (
        len(slice_shape) == 1
        and tensors[0].ndim == 1
        and inp_view.stride(-1) == 1
        and values.stride(-1) == 1
    )


def _launch(inp_view, tensors, values, slice_shape, accumulate):
    """Launch the capped kernel, or skip it when there is nothing to write.

    An empty index tensor makes the flattened grid 0, which the driver rejects
    with ``EE1003 ... coreDim ... expected: not equal to 0``; it is a no-op, so
    return without launching.
    """
    if tensors[0].numel() == 0 or math.prod(slice_shape) == 0:
        logger.debug("GEMS_ASCEND INDEX_PUT empty, nothing to write")
        return
    if not slice_shape and not accumulate:
        # Every dimension indexed: one element per lane, and the order plain
        # stores land in is not observable. The accumulate path keeps the
        # shared kernel, where the order its atomics land in is.
        _ascend_index_put_func(inp_view, tensors, values, accumulate, "scatter")
        return
    folded = _fold_trailing(inp_view, values, slice_shape)
    if _rowwise_eligible(folded[0], tensors, folded[1], folded[2]):
        _ascend_index_put_func(folded[0], tensors, folded[1], accumulate, "rowwise")
        return
    _ascend_index_put_func(inp_view, tensors, values, accumulate)


def index_put_(inp, indices, values, accumulate=False):
    logger.debug("GEMS_ASCEND INDEX_PUT_")
    inp_view, tensors, values, slice_shape = _preprocess(inp, indices, values)
    _launch(inp_view, tensors, values, slice_shape, accumulate)
    return inp


def index_put(inp, indices, values, accumulate=False):
    logger.debug("GEMS_ASCEND INDEX_PUT")
    return index_put_(inp.clone(), indices, values, accumulate)


def _index_put_impl_(inp, indices, values, accumulate=False, unsafe=False):
    logger.debug("GEMS_ASCEND _INDEX_PUT_IMPL_")

    # `unsafe` only tells the caller's autograd machinery that bounds have
    # already been checked; the kernel masks out-of-range indices either way.
    indices = list(indices)

    # A single bool mask selects `K` elements and the generic implementation
    # accepts a flat `values` of `K` entries for it, which plain broadcasting
    # would reject. Keep that leniency, the registered op has always had it.
    if len(indices) == 1 and indices[0] is not None and indices[0].dtype == torch.bool:
        mask = indices[0]
        if mask.device != inp.device:
            mask = mask.to(inp.device)
        where = list(torch.where(mask))
        k = where[0].numel()
        target_shape = (k,) + tuple(inp.shape[len(where) :])
        if values.numel() == 1:
            values = values.to(inp.device, inp.dtype).expand(target_shape)
        elif values.numel() == k and len(target_shape) == 1:
            # A mask over every dimension selects k scalars, and values is
            # allowed to arrive in any shape holding k of them.
            values = values.reshape(target_shape)
        indices = where

    inp_view, tensors, values, slice_shape = _preprocess(inp, indices, values)
    _launch(inp_view, tensors, values, slice_shape, accumulate)
    return inp
