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

import torch

logger = logging.getLogger(__name__)


def broadcast_to(x, size):
    """Ascend broadcast_to: a strided view, with no kernel launch.

    ``aten::broadcast_to`` is declared as
    ``broadcast_to(Tensor(a) self, SymInt[] size) -> Tensor(a)``. The ``(a)``
    on both sides says the result aliases its input, and torch resolves the op
    by rewriting shape and strides without reading the data. ``expand``, which
    FlagGems registers alongside this op and which the generic broadcast_to
    docstring calls it equivalent to, already resolves that way here.

    The generic implementation instead allocates the output and fills it with a
    gather kernel, paying a scalar load and an integer div/mod per dimension per
    element. That costs more than the op is worth on every shape, and on Ascend
    it additionally has to be squeezed past two hardware limits:

    * the driver flattens the launch grid into a uint16 ``coreDim`` before
      ``rtKernelLaunch``, so ``ceil(n_elements / 1024) > 65535`` - any output
      above 67108864 elements - is rejected with EE1003;
    * a kernel gets 192 KiB of unified buffer, which bounds the tile size.

    Neither limit applies to a view, and neither does Triton's lack of a complex
    pointer type (``canonicalize_dtype`` raises ``KeyError('complex64')``),
    torch_npu's eight-dimension cap, or the implicit host-to-device copy of the
    shape/stride arrays. Four other backends - thead, iluvatar, metax and
    kunlunxin - reached the same conclusion and dropped their kernels too.
    """
    logger.debug("GEMS_ASCEND BROADCAST_TO")

    if not isinstance(size, (list, tuple, torch.Size)):
        raise TypeError("broadcast_to size must be a list/tuple/torch.Size of ints")

    size = list(size)
    in_shape = list(x.shape)
    in_strides = list(x.stride())

    out_ndim = len(size)
    in_ndim = len(in_shape)

    if in_ndim > out_ndim:
        raise RuntimeError(
            f"broadcast_to: requested size has fewer dimensions ({out_ndim}) than input ({in_ndim})"
        )

    # Pad on the left so the stride list lines up with size. The dimensions that
    # appear here are new, so they are broadcast and carry stride 0.
    if in_ndim < out_ndim:
        pad = out_ndim - in_ndim
        in_shape = [1] * pad + in_shape
        in_strides = [0] * pad + in_strides

    out_shape = []
    for d in range(out_ndim):
        req = size[d]
        src = in_shape[d]
        target = src if req == -1 else req
        if src != target and src != 1:
            raise RuntimeError(
                f"The expanded size of the tensor ({target}) must match the existing size ({src}) "
                f"at non-singleton dimension {d}. Target sizes must be the same, or -1, "
                f"or the size of dimension in the original tensor must be 1."
            )
        out_shape.append(int(target))

    # Stride 0 on every dimension that is actually being broadcast, the source
    # stride everywhere else. A size-1 dimension that stays size 1 keeps its
    # own stride: nothing indexes it, but the value is visible through
    # .stride() and decides whether the result reports as contiguous, and
    # expand() keeps it. (The generic kernel feeds these strides to a gather
    # instead, where a size-1 axis contributes index 0 either way, so it can
    # and does zero them unconditionally.)
    out_strides = [
        0 if in_shape[d] == 1 and out_shape[d] != 1 else int(in_strides[d])
        for d in range(out_ndim)
    ]

    # as_strided addresses x's storage directly, so x's own shape stops
    # mattering here - but its storage_offset still does, or a sliced input
    # would be read from the start of the buffer it was sliced out of.
    return torch.as_strided(x, out_shape, out_strides, x.storage_offset())
