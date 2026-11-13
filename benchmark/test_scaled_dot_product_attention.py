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

from . import base


class AttentionBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (batch, num_heads, seq_len, head_size).
        return []


class AscendD256AttentionBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [(2, 4, 1, 32768, 256), (4, 6, 1, 8192, 256)]
    DEFAULT_SHAPE_DESC = "batch, query_heads, kv_heads, seq_len, head_dim"

    def set_shapes(self, shape_file_path=None):
        # This regression benchmark targets the two training workloads;
        # the generic Benchmark YAML entry uses incompatible tensor shapes.
        self.shapes = list(self.DEFAULT_SHAPES)
        self.shape_desc = self.DEFAULT_SHAPE_DESC


@pytest.mark.scaled_dot_product_attention_backward
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend D256 benchmark")
def test_scaled_dot_product_attention_ascend_d256_backward():
    import torch_npu

    def input_fn(shape, dtype, device):
        batch, heads, kv_heads, seq_len, head_dim = shape
        q, k, v = [
            torch.randn(batch, h, seq_len, head_dim, device=device, dtype=dtype)
            .mul_(0.5)
            .requires_grad_()
            for h in (heads, kv_heads, kv_heads)
        ]
        mask = torch.ones((2048, 2048), dtype=torch.bool).triu(1).to(device)
        yield q, k, v, mask

    def native(q, k, v, mask):
        return torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            q.shape[1],
            "BNSD",
            atten_mask=mask,
            scale=q.shape[-1] ** -0.5,
            keep_prob=1.0,
            pre_tockens=2147483647,
            next_tockens=0,
            sparse_mode=2,
        )[0]

    def gems(q, k, v, mask):
        return flag_gems.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True
        )

    # Benchmark builds the forward graph once and times all three gradients,
    # including preprocessing, bounded-cache launches and the GQA reduction.
    bench = AscendD256AttentionBenchmark(
        op_name="scaled_dot_product_attention_backward",
        input_fn=input_fn,
        torch_op=native,
        gems_op=gems,
        is_backward=True,
        dtypes=[torch.bfloat16, torch.float16],
    )
    bench.run()


def scaled_dot_product_flash_attention_input_fn(shape, dtype, device):
    query = torch.randn(shape, device=device, dtype=dtype)
    key = torch.randn(shape, device=device, dtype=dtype)
    value = torch.randn(shape, device=device, dtype=dtype)
    yield query, key, value, 0.0, False, False


@pytest.mark.scaled_dot_product_flash_attention
def test_scaled_dot_product_flash_attention():
    bench = AttentionBenchmark(
        op_name="scaled_dot_product_flash_attention",
        input_fn=scaled_dot_product_flash_attention_input_fn,
        torch_op=torch.ops.aten._scaled_dot_product_flash_attention.default,
        # FlashAttention supports CUDA float16 and bfloat16 inputs.
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()


@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize("dropout_p", [0.0])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_scaled_dot_product_attention(monkeypatch, dropout_p, is_causal):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    def scaled_dot_product_attention_kwargs(shape, dtype, device):
        query = torch.randn(shape, device=device, dtype=dtype)
        key = torch.randn(shape, device=device, dtype=dtype)
        value = torch.randn(shape, device=device, dtype=dtype)
        yield query, key, value, None, dropout_p, is_causal

    def sdpa_flash(
        query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=is_causal
    ):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    bench = AttentionBenchmark(
        op_name="scaled_dot_product_attention",
        input_fn=scaled_dot_product_attention_kwargs,
        # torch_op=torch.nn.functional.scaled_dot_product_attention,
        torch_op=sdpa_flash,
        gems_op=flag_gems.scaled_dot_product_attention,
        dtypes=[
            torch.float16,
            torch.bfloat16,
        ],
    )
    bench.run()


SDPA_FORWARD_SHAPES = [
    (4, 8, 1024, 64),
    (2, 4, 2048, 128),
    (2, 4, 512, 64),
]


@pytest.mark.scaled_dot_product_attention_forward
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("shape", SDPA_FORWARD_SHAPES)
def test_scaled_dot_product_attention_forward(shape, is_causal):
    batch, num_heads, seq_len, head_size = shape

    def input_fn(_shape, dtype, device):
        query = torch.randn(
            (batch, num_heads, seq_len, head_size), device=device, dtype=dtype
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        yield query, key, value, None, 0.0, is_causal

    bench = AttentionBenchmark(
        op_name="scaled_dot_product_attention_forward",
        input_fn=input_fn,
        torch_op=torch.nn.functional.scaled_dot_product_attention,
        gems_op=flag_gems.scaled_dot_product_attention_forward,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()
