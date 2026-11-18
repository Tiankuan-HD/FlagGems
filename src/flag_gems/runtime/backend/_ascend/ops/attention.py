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
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.runtime import driver

from flag_gems.config import use_c_extension
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import broadcasted_stride

from .flash_api import mha_fwd, mha_varlan_fwd, mha_varlan_fwd_opt

logger = logging.getLogger(__name__)


# Modified from Triton tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    query,  #
    K_block_ptr,
    V_block_ptr,  #
    mask_block_ptr,  #
    stride_k_seqlen,
    stride_v_seqlen,
    stride_attn_mask_kv_seqlen,  #
    start_m,
    qk_scale,  #
    q_load_mask,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    KV_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    HEAD_DIM_ACTUAL: tl.constexpr,
):
    # Align both causal regions to the same KV tile boundary. When N > M,
    # starting the masked region at the Q tile would overlap the last full
    # KV tile and expose future keys in the unmasked region.
    if STAGE == 1:
        lo, hi = 0, (start_m * BLOCK_M // BLOCK_N) * BLOCK_N
    elif STAGE == 2:
        lo = (start_m * BLOCK_M // BLOCK_N) * BLOCK_N
        hi = (start_m + 1) * BLOCK_M
    # causal = False
    else:
        lo, hi = 0, KV_CTX

    K_block_ptr += lo * stride_k_seqlen
    V_block_ptr += lo * stride_v_seqlen
    if HAS_ATTN_MASK:
        mask_block_ptr += lo * stride_attn_mask_kv_seqlen

    LOG2E = 1.44269504  # log2(e) constant
    head_mask = tl.arange(0, HEAD_DIM) < HEAD_DIM_ACTUAL

    # loop over key, value and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        kv_load_mask = (start_n + offs_n) < KV_CTX
        # start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        # Load along the contiguous head dimension before transposing for QK.
        # A directly loaded [D, N] tile lowers poorly on the Ascend backend.
        key = tl.trans(
            tl.load(
                K_block_ptr,
                mask=kv_load_mask[:, None] & head_mask[None, :],
                other=0.0,
            )
        )
        if PRE_LOAD_V:
            value = tl.load(
                V_block_ptr,
                mask=kv_load_mask[:, None] & head_mask[None, :],
                other=0.0,
            )

        qk = tl.dot(query, key, allow_tf32=False)
        # incase not divisible.
        qk = tl.where(kv_load_mask[None, :], qk, -float("inf"))
        # qk = qk.to(tl.float32)

        if HAS_ATTN_MASK:
            attn_mask = tl.load(
                mask_block_ptr,
                mask=q_load_mask[:, None] & kv_load_mask[None, :],
                other=0.0,
            )

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])

            if HAS_ATTN_MASK:
                qk = qk * qk_scale + attn_mask
                qk *= LOG2E
                qk = qk + tl.where(mask, 0, -1.0e6)
            else:
                qk = qk * qk_scale * LOG2E + tl.where(mask, 0, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            qk *= qk_scale * LOG2E
            if HAS_ATTN_MASK:
                qk = qk + attn_mask
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        if not PRE_LOAD_V:
            value = tl.load(
                V_block_ptr,
                mask=kv_load_mask[:, None] & head_mask[None, :],
                other=0.0,
            )
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(query.dtype)
        p = p.to(value.dtype)
        acc = tl.dot(p, value, acc, allow_tf32=False)
        # update m_i and l_i
        m_i = m_ij

        K_block_ptr += BLOCK_N * stride_k_seqlen
        V_block_ptr += BLOCK_N * stride_v_seqlen

        if HAS_ATTN_MASK:
            mask_block_ptr += BLOCK_N * stride_attn_mask_kv_seqlen

    return acc, l_i, m_i


@libentry()
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    attn_mask,
    sm_scale,
    M,
    Out,  #
    stride_q_batch,
    stride_q_head,
    stride_q_seqlen,
    stride_q_headsize,
    stride_k_batch,
    stride_k_head,
    stride_k_seqlen,
    stride_k_headsize,
    stride_v_batch,
    stride_v_head,
    stride_v_seqlen,
    stride_v_headsize,
    stride_attn_mask_batch,
    stride_attn_mask_head,
    stride_attn_mask_q_seqlen,
    stride_attn_mask_kv_seqlen,
    stride_o_batch,
    stride_o_head,
    stride_o_seqlen,
    stride_o_headsize,
    Z,
    q_head_num,
    kv_head_num,
    GROUP_HEAD: tl.constexpr,
    Q_CTX,
    KV_CTX,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_ACTUAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch_id = off_hz // q_head_num
    head_id = off_hz % q_head_num
    kv_head_id = head_id // GROUP_HEAD

    q_offset = (
        batch_id.to(tl.int64) * stride_q_batch + head_id.to(tl.int64) * stride_q_head
    )
    o_offset = (
        batch_id.to(tl.int64) * stride_o_batch + head_id.to(tl.int64) * stride_o_head
    )
    kv_offset = (
        batch_id.to(tl.int64) * stride_k_batch + kv_head_id.to(tl.int64) * stride_k_head
    )

    offs_headsize = tl.arange(0, HEAD_DIM)
    head_mask = offs_headsize < HEAD_DIM_ACTUAL

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_load_mask = offs_m < Q_CTX
    offs_n = tl.arange(0, BLOCK_N)

    Q_block_ptr = (
        Q
        + q_offset
        + offs_m[:, None] * stride_q_seqlen
        + offs_headsize[None, :] * stride_q_headsize
    )
    K_block_ptr = (
        K
        + kv_offset
        + offs_n[:, None] * stride_k_seqlen
        + offs_headsize[None, :] * stride_k_headsize
    )
    V_block_ptr = (
        V
        + kv_offset
        + offs_n[:, None] * stride_v_seqlen
        + offs_headsize[None, :] * stride_v_headsize
    )

    if HAS_ATTN_MASK:
        attn_mask_offset = (
            batch_id.to(tl.int64) * stride_attn_mask_batch
            + head_id.to(tl.int64) * stride_attn_mask_head
        )
        mask_block_ptr = (
            attn_mask
            + attn_mask_offset
            + offs_m[:, None] * stride_attn_mask_q_seqlen
            + offs_n[None, :] * stride_attn_mask_kv_seqlen
        )
    else:
        mask_block_ptr = None

    O_block_ptr = (
        Out
        + o_offset
        + offs_m[:, None] * stride_o_seqlen
        + offs_headsize[None, :] * stride_o_headsize
    )

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    # qk_scale *= 1.44269504  # 1/log(2)
    # load query: it will stay in SRAM throughout
    query = tl.load(
        Q_block_ptr, mask=q_load_mask[:, None] & head_mask[None, :], other=0.0
    )
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            query,
            K_block_ptr,
            V_block_ptr,
            mask_block_ptr,
            stride_k_seqlen,
            stride_v_seqlen,
            stride_attn_mask_kv_seqlen,
            start_m,
            qk_scale,
            q_load_mask,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            4 - STAGE,
            offs_m,
            offs_n,
            KV_CTX,
            V.dtype.element_ty == tl.float8e5,
            HAS_ATTN_MASK,
            PRE_LOAD_V,
            HEAD_DIM_ACTUAL,
        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            query,
            K_block_ptr,
            V_block_ptr,
            mask_block_ptr,
            stride_k_seqlen,
            stride_v_seqlen,
            stride_attn_mask_kv_seqlen,
            start_m,
            qk_scale,
            q_load_mask,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            KV_CTX,
            V.dtype.element_ty == tl.float8e5,
            HAS_ATTN_MASK,
            PRE_LOAD_V,
            HEAD_DIM_ACTUAL,
        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * Q_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=q_load_mask)
    tl.store(
        O_block_ptr,
        acc.to(Out.type.element_ty),
        mask=q_load_mask[:, None] & head_mask[None, :],
    )


@triton.jit
def _attn_bwd_preprocess(
    O_, DO, Delta, Z, H, Q_CTX, BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr
):  # noqa: E741
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = off_m < Q_CTX

    off_hz = tl.program_id(1)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(
        O_ + off_hz * D_HEAD * Q_CTX + off_m[:, None] * D_HEAD + off_n[None, :],
        mask=mask[:, None],
        other=0.0,
    )
    do = tl.load(
        DO + off_hz * D_HEAD * Q_CTX + off_m[:, None] * D_HEAD + off_n[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * Q_CTX + off_m, delta, mask=mask)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,  #
    Q,
    key,
    value,
    sm_scale,  #
    DO,  #
    M,
    D,  #
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    H,
    Q_CTX,
    KV_CTX,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,  #
    # Filled in by the wrapper.
    start_n,
    start_m,
    num_steps,  #
    MASK: tl.constexpr,
):
    # BLOCK_M1: 32
    # BLOCK_N1: 128
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_n_mask = offs_n < KV_CTX  # (BLOCK_N1, )

    offs_k = tl.arange(0, BLOCK_DMODEL)  # (BLOCK_DMODEL, )

    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        offs_m = curr_m + tl.arange(0, BLOCK_M1)  # (BLOCK_M1, )
        offs_m_mask = offs_m < Q_CTX  # (BLOCK_M1, )

        qT_ptrs = (
            Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
        )  # (BLOCK_DMODEL, BLOCK_M1)
        do_ptrs = (
            DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
        )  # (BLOCK_M1, BLOCK_DMODEL)

        qT = tl.load(
            qT_ptrs, mask=offs_m_mask[None, :], other=0.0
        )  # (BLOCK_DMODEL, BLOCK_M1)

        # Load m before computing qk to reduce pipeline stall.
        m = tl.load(M + offs_m, mask=offs_m_mask, other=float("inf"))  # (BLOCK_M1, )

        # key: (BLOCK_N1, BLOCK_DMODEL)
        qkT = tl.dot(key, qT)  # (BLOCK_N1, BLOCK_M1)
        m = tl.broadcast_to(m[None, :], (BLOCK_N1, BLOCK_M1))  # (BLOCK_N1, BLOCK_M1)
        m = tl.where(offs_n_mask[:, None], m, float("inf"))  # (BLOCK_N1, BLOCK_M1)
        pT = tl.math.exp2(qkT - m)
        # pT = tl.math.exp2(qkT - m[None, :])

        mask = (offs_m < Q_CTX)[None, :] & (offs_n < KV_CTX)[
            :, None
        ]  # (BLOCK_N1, BLOCK_M1)
        # Autoregressive masking.
        if MASK:
            mask &= offs_m[None, :] >= offs_n[:, None]
        pT = tl.where(mask, pT, 0.0)  # (BLOCK_N1, BLOCK_M1)

        do = tl.load(do_ptrs, mask=offs_m_mask[:, None], other=0.0)

        # Compute dV.
        dv += tl.dot(pT.to(do.dtype), do)  # (BLOCK_N1, BLOCK_DMODEL)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m, mask=offs_m_mask, other=0.0)  # (BLOCK_M1, )

        # Compute dP and dS.
        dpT = tl.dot(value, tl.trans(do)).to(
            tl.float32
        )  # (BLOCK_N1, BLOCK_DMODEL) @ (BLOCK_M1, BLOCK_DMODEL).T -> (BLOCK_N1, BLOCK_M1)
        dsT = pT * (dpT - Di[None, :])  # (BLOCK_N1, BLOCK_M1)
        dsT = dsT.to(qT.dtype)
        qT = tl.where(offs_m_mask[None, :], qT, 0.0)  # (BLOCK_DMODEL, BLOCK_M1)
        dsT = tl.where(
            offs_m_mask[None, :] & offs_n_mask[:, None], dsT, 0.0
        )  # (BLOCK_N1, BLOCK_M1)
        dk += tl.dot(
            dsT, tl.trans(qT)
        )  # (BLOCK_N1, BLOCK_M1) @ (BLOCK_DMODEL, BLOCK_M1).T -> (BLOCK_N1, BLOCK_DMODEL)
        # Increment pointers.
        curr_m += step_m
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    dq,
    query,
    K,
    V,  #
    do,
    m,
    D,
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    H,
    Q_CTX,  #
    KV_CTX,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,
    # Filled in by the wrapper.
    start_m,
    start_n,
    num_steps,  #
    MASK: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_m_mask = offs_m < Q_CTX

    offs_k = tl.arange(0, BLOCK_DMODEL)
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m, mask=offs_m_mask, other=0.0)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        offs_n_mask = offs_n < KV_CTX

        kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
        vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d

        kT = tl.load(kT_ptrs, mask=offs_n_mask[None, :], other=0.0)
        vT = tl.load(vT_ptrs, mask=offs_n_mask[None, :], other=0.0)
        qk = tl.dot(query, kT)
        p = tl.math.exp2(qk - m)
        mask = (offs_m < Q_CTX)[:, None] & (offs_n < KV_CTX)[None, :]
        # Autoregressive masking.
        if MASK:
            # mask = (offs_m[:, None] >= offs_n[None, :])
            # mask = (offs_m[:, None] >= offs_n[None, :]) & (offs_m < N_CTX)[:, None] & (offs_n < N_CTX)[None, :]
            mask &= offs_m[:, None] >= offs_n[None, :]
        p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = tl.where(mask, ds, 0.0).to(kT.dtype)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
    return dq


@libentry()
@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,  #
    DO,  #
    DQ,
    DK,
    DV,  #
    M,
    D,
    # shared by Q/K/V/DO.
    stride_z,
    stride_h,
    stride_tok,
    stride_d,  #
    kv_stride_z,
    kv_stride_h,  #
    H,  # query head num
    Q_CTX,  #
    KV_CTX,  #
    kv_head_num,  #
    GROUP_HEAD: tl.constexpr,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,
    COMPUTE_DQ: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    bhid = tl.program_id(2)
    batch_id = bhid // H
    q_head_id = bhid % H
    kv_head_id = q_head_id // GROUP_HEAD
    q_adj = (stride_h * q_head_id + stride_z * batch_id).to(tl.int64)
    kv_adj = (kv_stride_h * kv_head_id + kv_stride_z * batch_id).to(tl.int64)
    Q += q_adj
    K += kv_adj
    V += kv_adj
    DO += q_adj
    DQ += q_adj
    # dK/dV are temporarily expanded to query heads before GQA reduction.
    grad_kv_adj = (bhid * KV_CTX * BLOCK_DMODEL).to(tl.int64)
    DK += grad_kv_adj
    DV += grad_kv_adj
    M += (bhid * Q_CTX).to(tl.int64)
    D += (bhid * Q_CTX).to(tl.int64)
    pid = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # Compile dQ and dK/dV separately. Each uses one fully masked loop so
    # Ascend does not have to schedule nested diagonal/interior loop regions.
    if COMPUTE_DQ:
        start_m = pid * BLOCK_M2
        offs_m = start_m + tl.arange(0, BLOCK_M2)
        mask_m = offs_m < Q_CTX
        q_ptrs = offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
        query = tl.load(Q + q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(DO + q_ptrs, mask=mask_m[:, None], other=0.0)
        m = tl.load(M + offs_m, mask=mask_m, other=float("inf"))[:, None]
        dq = tl.zeros([BLOCK_M2, BLOCK_DMODEL], dtype=tl.float32)
        end_n = KV_CTX
        if IS_CAUSAL:
            end_n = tl.minimum(start_m + BLOCK_M2, KV_CTX)
        dq = _attn_bwd_dq(
            dq,
            query,
            K,
            V,
            do,
            m,
            D,
            stride_tok,
            stride_d,
            H,
            Q_CTX,
            KV_CTX,
            BLOCK_M2,
            BLOCK_N2,
            BLOCK_DMODEL,
            start_m,
            0,
            tl.cdiv(end_n, BLOCK_N2),
            MASK=IS_CAUSAL,
        )
        # K was pre-scaled by sm_scale * log2(e).
        dq *= 0.6931471805599453
        tl.store(DQ + q_ptrs, dq, mask=mask_m[:, None])
    else:
        start_n = pid * BLOCK_N1
        offs_n = start_n + tl.arange(0, BLOCK_N1)
        mask_n = offs_n < KV_CTX
        kv_ptrs = offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
        key = tl.load(K + kv_ptrs, mask=mask_n[:, None], other=0.0)
        value = tl.load(V + kv_ptrs, mask=mask_n[:, None], other=0.0)
        dk = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
        start_m = 0
        if IS_CAUSAL:
            start_m = start_n
        num_steps = tl.cdiv(tl.maximum(Q_CTX - start_m, 0), BLOCK_M1)
        dk, dv = _attn_bwd_dkdv(
            dk,
            dv,
            Q,
            key,
            value,
            sm_scale,
            DO,
            M,
            D,
            stride_tok,
            stride_d,
            H,
            Q_CTX,
            KV_CTX,
            BLOCK_M1,
            BLOCK_N1,
            BLOCK_DMODEL,
            start_n,
            start_m,
            num_steps,
            MASK=IS_CAUSAL,
        )
        if IS_CAUSAL:
            # Fully masked KV tiles have a zero-trip loop. Explicitly zero
            # their gradients: Ascend may leave the dot accumulators stale
            # when the loop does not execute.
            has_query = offs_n[:, None] < Q_CTX
            dk = tl.where(has_query, dk, 0.0)
            dv = tl.where(has_query, dv, 0.0)
        tl.store(DK + kv_ptrs, dk * sm_scale, mask=mask_n[:, None])
        tl.store(DV + kv_ptrs, dv, mask=mask_n[:, None])


def _use_aligned_d256(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa):
    """The mask-free loads below require equal, 128-aligned sequence lengths.

    Keep irregular lengths, other dtypes, masks and noncausal attention on the
    established Ascend implementation. FP32 causal comparisons are exact within
    the supported sequence range. This dispatch never calls the vendor backend.
    """
    if attn_mask is not None or dropout_p != 0.0 or not is_causal:
        return False
    if any(
        x.ndim != 4
        or x.dtype not in (torch.bfloat16, torch.float16)
        or not x.is_contiguous()
        for x in (query, key, value)
    ):
        return False
    b, h, s, d = query.shape
    if b <= 0 or h <= 0 or d != 256 or s < 128 or s % 128 or s > 2**24:
        return False
    if (
        key.dtype != query.dtype
        or value.dtype != query.dtype
        or key.shape != value.shape
        or key.shape[0] != b
        or key.shape[2:] != (s, d)
    ):
        return False
    kh = key.shape[1]
    return kh > 0 and h % kh == 0 and (enable_gqa or h == kh)


# Modified from Triton tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
@triton.jit
def _attn_fwd_d256_inner(
    acc,
    l_i,
    m_i,
    query,  #
    K_block_ptr,
    V_block_ptr,  #
    mask_block_ptr,  #
    stride_k_seqlen: tl.constexpr,
    stride_v_seqlen: tl.constexpr,
    stride_attn_mask_kv_seqlen: tl.constexpr,  #
    start_m,
    qk_scale,  #
    q_load_mask,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    KV_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
):
    # Align both causal regions to the same KV tile boundary. When N > M,
    # starting the masked region at the Q tile would overlap the last full
    # KV tile and expose future keys in the unmasked region.
    if STAGE == 1:
        lo, hi = 0, (start_m * BLOCK_M // BLOCK_N) * BLOCK_N
    elif STAGE == 2:
        lo = (start_m * BLOCK_M // BLOCK_N) * BLOCK_N
        hi = (start_m + 1) * BLOCK_M
    # causal = False
    else:
        lo, hi = 0, KV_CTX

    K_block_ptr += lo * stride_k_seqlen
    V_block_ptr += lo * stride_v_seqlen
    if HAS_ATTN_MASK:
        mask_block_ptr += lo * stride_attn_mask_kv_seqlen

    LOG2E = 1.44269504  # log2(e) constant

    # loop over key, value and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        kv_load_mask = tl.full((BLOCK_N,), True, tl.int1)
        # start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        # Load along the contiguous head dimension before transposing for QK.
        # A directly loaded [D, N] tile lowers poorly on the Ascend backend.
        key = tl.trans(tl.load(K_block_ptr, mask=kv_load_mask[:, None], other=0.0))
        if PRE_LOAD_V:
            value = tl.load(V_block_ptr, mask=kv_load_mask[:, None], other=0.0)

        qk = tl.dot(query, key, allow_tf32=False)
        # incase not divisible.
        qk = tl.where(kv_load_mask[None, :], qk, -float("inf"))
        # qk = qk.to(tl.float32)

        if HAS_ATTN_MASK:
            attn_mask = tl.load(
                mask_block_ptr,
                mask=q_load_mask[:, None] & kv_load_mask[None, :],
                other=0.0,
            )

        if STAGE == 2:
            mask = offs_m[:, None].to(tl.float32) >= (start_n + offs_n[None, :]).to(
                tl.float32
            )

            if HAS_ATTN_MASK:
                qk = qk * qk_scale + attn_mask
                qk *= LOG2E
                qk = qk + tl.where(mask, 0, -1.0e6)
            else:
                qk = qk * qk_scale * LOG2E + tl.where(mask, 0, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            qk *= qk_scale * LOG2E
            if HAS_ATTN_MASK:
                qk = qk + attn_mask
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

        p = tl.math.exp2(qk)
        # Preserve FP32 probabilities for the online normalization sum.
        p_full = p
        if not PRE_LOAD_V:
            value = tl.load(V_block_ptr, mask=kv_load_mask[:, None], other=0.0)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(query.dtype)
        p = p.to(value.dtype)
        # Start PV before the independent vector updates to the softmax state.
        pv = tl.dot(p, value, allow_tf32=False)
        l_ij = tl.sum(p_full, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + pv
        # update m_i and l_i
        m_i = m_ij

        K_block_ptr += BLOCK_N * stride_k_seqlen
        V_block_ptr += BLOCK_N * stride_v_seqlen

        if HAS_ATTN_MASK:
            mask_block_ptr += BLOCK_N * stride_attn_mask_kv_seqlen

    return acc, l_i, m_i


@libentry()
@triton.jit
def _attn_fwd_d256(
    Q,
    K,
    V,
    attn_mask,
    sm_scale: tl.constexpr,
    M,
    Out,  #
    stride_q_batch: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_seqlen: tl.constexpr,
    stride_q_headsize: tl.constexpr,
    stride_k_batch: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_k_seqlen: tl.constexpr,
    stride_k_headsize: tl.constexpr,
    stride_v_batch: tl.constexpr,
    stride_v_head: tl.constexpr,
    stride_v_seqlen: tl.constexpr,
    stride_v_headsize: tl.constexpr,
    stride_attn_mask_batch: tl.constexpr,
    stride_attn_mask_head: tl.constexpr,
    stride_attn_mask_q_seqlen: tl.constexpr,
    stride_attn_mask_kv_seqlen: tl.constexpr,
    stride_o_batch: tl.constexpr,
    stride_o_head: tl.constexpr,
    stride_o_seqlen: tl.constexpr,
    stride_o_headsize: tl.constexpr,
    Z: tl.constexpr,
    q_head_num: tl.constexpr,
    kv_head_num: tl.constexpr,
    GROUP_HEAD: tl.constexpr,
    Q_CTX: tl.constexpr,
    KV_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch_id = off_hz // q_head_num
    head_id = off_hz % q_head_num
    kv_head_id = head_id // GROUP_HEAD

    q_offset = (
        batch_id.to(tl.int64) * stride_q_batch + head_id.to(tl.int64) * stride_q_head
    )
    o_offset = (
        batch_id.to(tl.int64) * stride_o_batch + head_id.to(tl.int64) * stride_o_head
    )
    kv_offset = (
        batch_id.to(tl.int64) * stride_k_batch + kv_head_id.to(tl.int64) * stride_k_head
    )

    offs_headsize = tl.arange(0, HEAD_DIM)

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_load_mask = tl.full((BLOCK_M,), True, tl.int1)
    offs_n = tl.arange(0, BLOCK_N)

    Q_block_ptr = (
        Q
        + q_offset
        + offs_m[:, None] * stride_q_seqlen
        + offs_headsize[None, :] * stride_q_headsize
    )
    K_block_ptr = (
        K
        + kv_offset
        + offs_n[:, None] * stride_k_seqlen
        + offs_headsize[None, :] * stride_k_headsize
    )
    V_block_ptr = (
        V
        + kv_offset
        + offs_n[:, None] * stride_v_seqlen
        + offs_headsize[None, :] * stride_v_headsize
    )

    if HAS_ATTN_MASK:
        attn_mask_offset = (
            batch_id.to(tl.int64) * stride_attn_mask_batch
            + head_id.to(tl.int64) * stride_attn_mask_head
        )
        mask_block_ptr = (
            attn_mask
            + attn_mask_offset
            + offs_m[:, None] * stride_attn_mask_q_seqlen
            + offs_n[None, :] * stride_attn_mask_kv_seqlen
        )
    else:
        mask_block_ptr = None

    O_block_ptr = (
        Out
        + o_offset
        + offs_m[:, None] * stride_o_seqlen
        + offs_headsize[None, :] * stride_o_headsize
    )

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    # qk_scale *= 1.44269504  # 1/log(2)
    # load query: it will stay in SRAM throughout
    query = tl.load(Q_block_ptr, mask=q_load_mask[:, None], other=0.0)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_d256_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_d256_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_d256_inner(
            acc,
            l_i,
            m_i,
            query,
            K_block_ptr,
            V_block_ptr,
            mask_block_ptr,
            stride_k_seqlen,
            stride_v_seqlen,
            stride_attn_mask_kv_seqlen,
            start_m,
            qk_scale,
            q_load_mask,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            4 - STAGE,
            offs_m,
            offs_n,
            KV_CTX,
            V.dtype.element_ty == tl.float8e5,
            HAS_ATTN_MASK,
            PRE_LOAD_V,
        )
    # stage 2: on-band
    if STAGE & 2:
        # Apply the causal mask only in the diagonal KV region.
        acc, l_i, m_i = _attn_fwd_d256_inner(
            acc,
            l_i,
            m_i,
            query,
            K_block_ptr,
            V_block_ptr,
            mask_block_ptr,
            stride_k_seqlen,
            stride_v_seqlen,
            stride_attn_mask_kv_seqlen,
            start_m,
            qk_scale,
            q_load_mask,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            KV_CTX,
            V.dtype.element_ty == tl.float8e5,
            HAS_ATTN_MASK,
            PRE_LOAD_V,
        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * Q_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=q_load_mask)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=q_load_mask[:, None])


@triton.jit
def _attn_bwd_d256_preprocess(
    O_, DO, Delta, Z, H, Q_CTX, BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr
):  # noqa: E741
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = off_m < Q_CTX

    off_hz = tl.program_id(1)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(
        O_ + off_hz * D_HEAD * Q_CTX + off_m[:, None] * D_HEAD + off_n[None, :],
        mask=mask[:, None],
        other=0.0,
    )
    do = tl.load(
        DO + off_hz * D_HEAD * Q_CTX + off_m[:, None] * D_HEAD + off_n[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * Q_CTX + off_m, delta, mask=mask)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_d256_dkdv(
    dk,
    dv,  #
    GRAD_KV: tl.constexpr,
    Q,
    key,
    value,
    sm_scale: tl.constexpr,  #
    DO,  #
    M,
    D,  #
    # shared by Q/K/V/DO.
    stride_tok: tl.constexpr,
    stride_d: tl.constexpr,  #
    H: tl.constexpr,
    Q_CTX: tl.constexpr,
    KV_CTX: tl.constexpr,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,  #
    # Filled in by the wrapper.
    start_n,
    start_m,
    num_steps,  #
    MASK: tl.constexpr,
):
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    offs_k = tl.arange(0, BLOCK_DMODEL)  # (BLOCK_DMODEL, )

    # The single masked loop also supports M > N.
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        offs_m = curr_m + tl.arange(0, BLOCK_M1)  # (BLOCK_M1, )
        offs_m_mask = tl.full((BLOCK_M1,), True, tl.int1)  # (BLOCK_M1, )

        qT_ptrs = (
            Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
        )  # (BLOCK_DMODEL, BLOCK_M1)
        do_ptrs = (
            DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
        )  # (BLOCK_M1, BLOCK_DMODEL)

        qT = tl.load(
            qT_ptrs, mask=offs_m_mask[None, :], other=0.0
        )  # (BLOCK_DMODEL, BLOCK_M1)

        # Load m before computing qk to reduce pipeline stall.
        m = tl.load(M + offs_m, mask=offs_m_mask, other=float("inf"))  # (BLOCK_M1, )

        # key: (BLOCK_N1, BLOCK_DMODEL)
        if GRAD_KV == 1:
            qkT = tl.dot(tl.trans(qT), tl.trans(key))
        else:
            qkT = tl.dot(key, qT)
        if GRAD_KV == 2:
            # Fold the two scalar factors at compile time for dV. Keep the
            # probability reconstruction and accumulation in FP32.
            qkT = qkT * (sm_scale * 1.44269504)
        else:
            qkT = qkT * sm_scale * 1.44269504  # (BLOCK_N1, BLOCK_M1)
        if GRAD_KV == 1:
            m = m[:, None]
        else:
            m = m[None, :]
        pT = tl.math.exp2(qkT - m)
        # pT = tl.math.exp2(qkT - m[None, :])

        if GRAD_KV == 1:
            mask = tl.full((BLOCK_M1, BLOCK_N1), True, tl.int1)
            if MASK:
                mask &= offs_m[:, None].to(tl.float32) >= offs_n[None, :].to(tl.float32)
        else:
            mask = tl.full((BLOCK_N1, BLOCK_M1), True, tl.int1)
            if MASK:
                mask &= offs_m[None, :].to(tl.float32) >= offs_n[:, None].to(tl.float32)
        pT = tl.where(mask, pT, 0.0)

        do = tl.load(do_ptrs, mask=offs_m_mask[:, None], other=0.0)

        # Compute dV.
        if GRAD_KV != 1:
            dv += tl.dot(pT.to(do.dtype), do)  # (BLOCK_N1, BLOCK_DMODEL)
        if GRAD_KV != 2:
            # D is the per-query-row dot product of O and dO.
            Di = tl.load(D + offs_m, mask=offs_m_mask, other=0.0)  # (BLOCK_M1, )

            # dK uses query-major scores and a feature-major FP32 accumulator.
            if GRAD_KV == 1:
                dp = tl.dot(do, tl.trans(value))
                ds = (pT * (dp - Di[:, None])).to(qT.dtype)
                dk += tl.dot(qT, ds)
            else:
                dpT = tl.dot(value, tl.trans(do))
                dsT = (pT * (dpT - Di[None, :])).to(qT.dtype)
                dk += tl.dot(qT, tl.trans(dsT))
        # Increment pointers.
        curr_m += step_m
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_d256_dq(
    dq,
    query,
    K,
    V,  #
    sm_scale: tl.constexpr,
    do,
    m,
    D,
    # shared by Q/K/V/DO.
    stride_tok: tl.constexpr,
    stride_d: tl.constexpr,  #
    H: tl.constexpr,
    Q_CTX: tl.constexpr,  #
    KV_CTX: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,
    # Filled in by the wrapper.
    start_m,
    start_n,
    num_steps,  #
    MASK: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_m_mask = tl.full((BLOCK_M2,), True, tl.int1)

    offs_k = tl.arange(0, BLOCK_DMODEL)
    # D is the per-query-row dot product of O and dO.
    Di = tl.load(D + offs_m, mask=offs_m_mask, other=0.0)
    # The fully masked loop also supports N > M.
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        offs_n_mask = tl.full((BLOCK_N2,), True, tl.int1)

        kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
        vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d

        kT = tl.load(kT_ptrs, mask=offs_n_mask[None, :], other=0.0)
        vT = tl.load(vT_ptrs, mask=offs_n_mask[None, :], other=0.0)
        qk = tl.dot(tl.trans(kT), tl.trans(query))
        qk = qk * sm_scale * 1.44269504
        p = tl.math.exp2(qk - tl.trans(m))
        mask = (
            tl.full((offs_m.shape[0],), True, tl.int1)[None, :]
            & tl.full((offs_n.shape[0],), True, tl.int1)[:, None]
        )
        # Autoregressive masking.
        if MASK:
            mask &= offs_m[None, :].to(tl.float32) >= offs_n[:, None].to(tl.float32)
        p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(tl.trans(vT), tl.trans(do)).to(tl.float32)
        ds = p * (dp - Di[None, :])
        ds = tl.where(mask, ds, 0.0).to(kT.dtype)
        # Compute dQ.
        # K stays in its original dtype; scale the FP32 gradient at the end.
        dq += tl.dot(kT, ds)
        # Increment pointers.
        curr_n += step_n
    return dq


@libentry()
@triton.jit
def _attn_bwd_d256(
    Q,
    K,
    V,
    sm_scale: tl.constexpr,  #
    DO,  #
    DQ,
    DK,
    DV,  #
    M,
    D,
    # shared by Q/K/V/DO.
    stride_z: tl.constexpr,
    stride_h: tl.constexpr,
    stride_tok: tl.constexpr,
    stride_d: tl.constexpr,  #
    kv_stride_z: tl.constexpr,
    kv_stride_h: tl.constexpr,  #
    H: tl.constexpr,  # query head num
    Q_CTX: tl.constexpr,  #
    KV_CTX: tl.constexpr,  #
    kv_head_num: tl.constexpr,  #
    GROUP_HEAD: tl.constexpr,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,
    COMPUTE_DQ: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    GRAD_KV: tl.constexpr = 0,
):
    # Schedule consecutive sequence tiles across cores for dK/dV; retain
    # the original head-first order for dQ.
    bhid = tl.program_id(2) if COMPUTE_DQ else tl.program_id(0)
    batch_id = bhid // H
    q_head_id = bhid % H
    kv_head_id = q_head_id // GROUP_HEAD
    q_adj = (stride_h * q_head_id + stride_z * batch_id).to(tl.int64)
    kv_adj = (kv_stride_h * kv_head_id + kv_stride_z * batch_id).to(tl.int64)
    Q += q_adj
    K += kv_adj
    V += kv_adj
    DO += q_adj
    DQ += q_adj
    # dK/dV are temporarily expanded to query heads before GQA reduction.
    grad_kv_adj = (bhid * KV_CTX * BLOCK_DMODEL).to(tl.int64)
    DK += grad_kv_adj
    DV += grad_kv_adj
    M += (bhid * Q_CTX).to(tl.int64)
    D += (bhid * Q_CTX).to(tl.int64)
    pid = tl.program_id(0) if COMPUTE_DQ else tl.program_id(2)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # Keep dK/dQ in a single masked loop. dV peels its first causal tile
    # below so the remaining query tiles need no elementwise causal mask.
    if COMPUTE_DQ:
        start_m = pid * BLOCK_M2
        offs_m = start_m + tl.arange(0, BLOCK_M2)
        mask_m = tl.full((BLOCK_M2,), True, tl.int1)
        q_ptrs = offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
        query = tl.load(Q + q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(DO + q_ptrs, mask=mask_m[:, None], other=0.0)
        m = tl.load(M + offs_m, mask=mask_m, other=float("inf"))[:, None]
        # Feature-major accumulation avoids a transpose in each loop iteration.
        dq = tl.zeros([BLOCK_DMODEL, BLOCK_M2], dtype=tl.float32)
        end_n = KV_CTX
        if IS_CAUSAL:
            end_n = tl.minimum(start_m + BLOCK_M2, KV_CTX)
        dq = _attn_bwd_d256_dq(
            dq,
            query,
            K,
            V,
            sm_scale,
            do,
            m,
            D,
            stride_tok,
            stride_d,
            H,
            Q_CTX,
            KV_CTX,
            BLOCK_M2,
            BLOCK_N2,
            BLOCK_DMODEL,
            start_m,
            0,
            tl.cdiv(end_n, BLOCK_N2),
            MASK=IS_CAUSAL,
        )
        # Apply the softmax scale after accumulating against unscaled K.
        dq *= sm_scale
        tl.store(DQ + q_ptrs, tl.trans(dq), mask=mask_m[:, None])
    else:
        start_n = pid * BLOCK_N1
        offs_n = start_n + tl.arange(0, BLOCK_N1)
        mask_n = tl.full((BLOCK_N1,), True, tl.int1)
        kv_ptrs = offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
        key = tl.load(K + kv_ptrs, mask=mask_n[:, None], other=0.0)
        value = tl.load(V + kv_ptrs, mask=mask_n[:, None], other=0.0)
        # dK is feature-major; dV retains its sequence-major accumulator.
        dk = tl.zeros([BLOCK_DMODEL, BLOCK_N1], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
        start_m = 0
        if IS_CAUSAL:
            start_m = (start_n // BLOCK_M1) * BLOCK_M1
        num_steps = tl.cdiv(tl.maximum(Q_CTX - start_m, 0), BLOCK_M1)
        # Only the first query tile intersects the causal boundary.
        # Keep dV's interior loop free of per-element causal comparisons.
        if GRAD_KV == 2 and IS_CAUSAL:
            tl.static_assert(BLOCK_M1 >= BLOCK_N1)
            tl.static_assert(BLOCK_M1 % BLOCK_N1 == 0)
            dk, dv = _attn_bwd_d256_dkdv(
                dk,
                dv,
                GRAD_KV,
                Q,
                key,
                value,
                sm_scale,
                DO,
                M,
                D,
                stride_tok,
                stride_d,
                H,
                Q_CTX,
                KV_CTX,
                BLOCK_M1,
                BLOCK_N1,
                BLOCK_DMODEL,
                start_n,
                start_m,
                1,
                MASK=IS_CAUSAL,
            )
            dk, dv = _attn_bwd_d256_dkdv(
                dk,
                dv,
                GRAD_KV,
                Q,
                key,
                value,
                sm_scale,
                DO,
                M,
                D,
                stride_tok,
                stride_d,
                H,
                Q_CTX,
                KV_CTX,
                BLOCK_M1,
                BLOCK_N1,
                BLOCK_DMODEL,
                start_n,
                start_m + BLOCK_M1,
                num_steps - 1,
                MASK=False,
            )
        else:
            dk, dv = _attn_bwd_d256_dkdv(
                dk,
                dv,
                GRAD_KV,
                Q,
                key,
                value,
                sm_scale,
                DO,
                M,
                D,
                stride_tok,
                stride_d,
                H,
                Q_CTX,
                KV_CTX,
                BLOCK_M1,
                BLOCK_N1,
                BLOCK_DMODEL,
                start_n,
                start_m,
                num_steps,
                MASK=IS_CAUSAL,
            )
        if GRAD_KV != 2:
            tl.store(DK + kv_ptrs, tl.trans(dk) * sm_scale, mask=mask_n[:, None])
        if GRAD_KV != 1:
            tl.store(DV + kv_ptrs, dv, mask=mask_n[:, None])


@triton.jit
def _query_tile(worker, step, Q_TILES: tl.constexpr, WORKERS: tl.constexpr):
    pair = worker + (step // 2) * WORKERS
    head = pair // (Q_TILES // 2)
    short = pair % (Q_TILES // 2)
    row = tl.where(step % 2 == 0, Q_TILES - 1 - short, short)
    return head, row


@triton.jit
def _publish_qk(q, key_ptr, qk_workspace, tile, width: tl.constexpr):
    rc = tl.arange(0, width)
    rm = tl.arange(0, 64)
    key = tl.load(key_ptr + (tile * width + rc[:, None]) * 256 + tl.arange(0, 256)[None, :])
    qk = tl.dot(q, tl.trans(key), allow_tf32=False)
    slot = tile % 2
    tl.store(qk_workspace + slot * 64 * width + rm[:, None] * width + rc[None, :], qk)
    if slot == 0:
        al.sync_block_set("cube", "vector", 0)
    else:
        al.sync_block_set("cube", "vector", 1)


@triton.jit
def _publish_pv(value_ptr, p_workspace, pv_workspace, tile, width: tl.constexpr):
    rc = tl.arange(0, width)
    rd = tl.arange(0, 256)
    rm = tl.arange(0, 64)
    value = tl.load(value_ptr + (tile * width + rc[:, None]) * 256 + rd[None, :])
    slot = tile % 2
    # Mode 2 credits the Cube after both Vector shards publish this P slot.
    if slot == 0:
        al.sync_block_wait("vector", "cube", 3)
    else:
        al.sync_block_wait("vector", "cube", 4)
    p = tl.load(p_workspace + (slot * 64 + rm[:, None]) * width + rc[None, :])
    pv = tl.dot(p, value, allow_tf32=False)
    tl.store(pv_workspace + (slot * 64 + rm[:, None]) * 256 + rd[None, :], pv)
    if slot == 0:
        al.sync_block_set("cube", "vector", 2)
    else:
        al.sync_block_set("cube", "vector", 5)


@triton.jit
def _publish_probability(
    qk_workspace, p_workspace, tile, row, lane, maximum, total, last,
    scale, width: tl.constexpr,
):
    rc = tl.arange(0, width)
    rm = lane * 32 + tl.arange(0, 32)
    slot = tile % 2
    if slot == 0:
        al.sync_block_wait("cube", "vector", 0)
    else:
        al.sync_block_wait("cube", "vector", 1)
    qk = tl.load(qk_workspace + slot * 64 * width + rm[:, None] * width + rc[None, :])
    scores = qk * scale
    if tile == last:
        q_pos = row * 64 + rm
        k_pos = tile * width + rc
        scores = tl.where(q_pos[:, None] >= k_pos[None, :], scores, -float("inf"))
    max_new = tl.maximum(maximum, tl.max(scores, 1))
    probability = tl.exp2(scores - max_new[:, None])
    alpha = tl.exp2(maximum - max_new)
    tl.store(
        p_workspace + (slot * 64 + rm[:, None]) * width + rc[None, :],
        probability.to(p_workspace.dtype.element_ty),
    )
    if slot == 0:
        al.sync_block_set("vector", "cube", 3)
    else:
        al.sync_block_set("vector", "cube", 4)
    total = total * alpha + tl.sum(probability, 1)
    return max_new, total, alpha


@triton.jit
def _accumulate_pv(pv_workspace, tile, lane, acc, alpha):
    rm = lane * 32 + tl.arange(0, 32)
    rd = tl.arange(0, 256)
    slot = tile % 2
    scaled_acc = acc * alpha[:, None]
    if slot == 0:
        al.sync_block_wait("cube", "vector", 2)
    else:
        al.sync_block_wait("cube", "vector", 5)
    pv = tl.load(pv_workspace + (slot * 64 + rm[:, None]) * 256 + rd[None, :])
    return scaled_acc + pv


@libentry()
@triton.jit
def _attn_fwd_d256_cv(
    Q, K, V, Out, Lse, QKTemp, PTemp, PVTemp, scale,
    S: tl.constexpr, HEADS: tl.constexpr, KV_HEADS: tl.constexpr, BATCH: tl.constexpr,
    WORKERS: tl.constexpr, KV_WIDTH: tl.constexpr,
):
    Q_TILES: tl.constexpr = S // 64
    PAIRS: tl.constexpr = BATCH * HEADS * (Q_TILES // 2)
    worker = tl.program_id(0)
    task_count = 2 * ((PAIRS + WORKERS - 1 - worker) // WORKERS)
    qk_workspace = QKTemp + worker * (2 * 64 * KV_WIDTH)
    p_workspace = PTemp + worker * (2 * 64 * KV_WIDTH)
    pv_workspace = PVTemp + worker * (2 * 64 * 256)
    dim = tl.arange(0, 256)

    with al.scope(core_mode="cube"):
        for step in range(task_count):
            head, row = _query_tile(worker, step, Q_TILES, WORKERS)
            kv_head = head // HEADS * KV_HEADS + head % HEADS // (HEADS // KV_HEADS)
            q_rows = row * 64 + tl.arange(0, 64)
            q = tl.load(Q + (head.to(tl.int64) * S + q_rows[:, None]) * 256 + dim[None, :])
            key_ptr = K + kv_head.to(tl.int64) * S * 256
            value_ptr = V + kv_head.to(tl.int64) * S * 256
            last = row * 64 // KV_WIDTH

            _publish_qk(q, key_ptr, qk_workspace, 0, KV_WIDTH)
            for tile in range(last):
                # Dispatch QK(t+1) before waiting for either half of P(t).
                # P(t-1) was already consumed, so the old QK slot is free.
                _publish_qk(q, key_ptr, qk_workspace, tile + 1, KV_WIDTH)
                _publish_pv(value_ptr, p_workspace, pv_workspace, tile, KV_WIDTH)
            _publish_pv(value_ptr, p_workspace, pv_workspace, last, KV_WIDTH)

    with al.scope(core_mode="vector"):
        tl.static_assert(al.sub_vec_num() == 2)
        lane = al.sub_vec_id().to(tl.int32)
        for step in range(task_count):
            head, row = _query_tile(worker, step, Q_TILES, WORKERS)
            last = row * 64 // KV_WIDTH
            max_score = tl.full((32,), -float("inf"), tl.float32)
            total = tl.full((32,), 1.0, tl.float32)
            acc = tl.full((32, 256), 0.0, tl.float32)
            alpha = tl.full((32,), 0.0, tl.float32)
            # Start by publishing P(0); finish by draining PV(last).
            # Both Vector shards read PV(t-1) before publishing P(t+1).
            for tile in range(last + 2):
                alpha_next = alpha
                if tile <= last:
                    max_score, total, alpha_next = _publish_probability(
                        qk_workspace, p_workspace, tile, row, lane,
                        max_score, total, last, scale, KV_WIDTH,
                    )
                if tile > 0:
                    acc = _accumulate_pv(pv_workspace, tile - 1, lane, acc, alpha)
                alpha = alpha_next
            rows = head.to(tl.int64) * S + row * 64 + lane * 32 + tl.arange(0, 32)
            tl.store(Out + rows[:, None] * 256 + dim[None, :], acc / total[:, None])
            tl.store(Lse + rows, max_score + tl.log2(total))


def _d256_causal_forward_cv(query, key, value, scale=None):
    """Requires aligned contiguous FP16/BF16 BNSD and two Vector subcores."""
    batch, heads, seq, dim = query.shape
    kv_heads = key.shape[1]
    if (
        batch <= 0 or dim != 256 or seq < 128 or seq % 128 or seq > 2**24
        or heads <= 0 or kv_heads <= 0 or heads % kv_heads
        or key.shape != (batch, kv_heads, seq, dim) or value.shape != key.shape
        or any(
            x.dtype not in (torch.bfloat16, torch.float16)
            or x.dtype != query.dtype or not x.is_contiguous() or x.device != query.device
            for x in (query, key, value)
        )
    ):
        raise ValueError("The explicit D256 forward requires aligned FP16/BF16 BNSD Q/K/V")
    width = 256 if seq % 256 == 0 else 128
    with torch_device_fn.device(query.device):
        properties = driver.active.utils.get_device_properties(query.device.index)
    workers = min(properties["num_aicore"], batch * heads * (seq // 128))
    score_scale = (1.0 / 16 if scale is None else scale) * 1.44269504
    output = torch.empty_like(query)
    lse = torch.empty((batch, heads, seq), device=query.device, dtype=torch.float32)
    qk = torch.empty((workers, 2, 64, width), device=query.device, dtype=torch.float32)
    p = torch.empty((workers, 2, 64, width), device=query.device, dtype=query.dtype)
    pv = torch.empty((workers, 2, 64, dim), device=query.device, dtype=torch.float32)
    with torch_device_fn.device(query.device):
        _attn_fwd_d256_cv[(workers,)](
            query, key, value, output, lse, qk, p, pv, score_scale,
            seq, heads, kv_heads, batch, workers, width,
            num_warps=4,
            num_stages=1,
            enable_auto_bind_sub_block=False,
            disable_auto_inject_block_sync=True,
            sync_solver=False,
            enable_ubuf_saving=True,
        )
    return output, lse


def _aligned_d256_forward(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    logger.debug("GEMS_ASCEND ALIGNED_D256_ATTENTION_FORWARD")
    if (
        _use_aligned_d256(
            query, key, value, attn_mask, dropout_p, is_causal, enable_gqa
        )
        and (scale is None or isinstance(scale, (int, float)) and math.isfinite(scale))
    ):
        return _d256_causal_forward_cv(query, key, value, scale=scale)
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    assert dropout_p == 0.0, "Currenty only support dropout_p=0.0"

    o = torch.empty_like(query, dtype=value.dtype)

    stage = 3 if is_causal else 1

    if scale is None:
        sm_scale = 1.0 / (HEAD_DIM_K**0.5)
    else:
        sm_scale = scale

    q_head_num = query.shape[1]
    kv_head_num = key.shape[1]
    assert enable_gqa or q_head_num == kv_head_num, (
        f"q_head_num {q_head_num} != kv_head_num {kv_head_num}, "
        "enable_gqa must be True to support different head numbers."
    )

    grid = lambda args: (
        triton.cdiv(query.shape[2], args["BLOCK_M"]),
        query.shape[0] * query.shape[1],
        1,
    )

    if attn_mask is not None:
        HAS_ATTN_MASK = True
        if attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.to(query.dtype) * -1.0e6
        stride_attn_mask_batch = attn_mask.stride(0)
        stride_attn_mask_head = attn_mask.stride(1)
        stride_attn_mask_q_seqlen = attn_mask.stride(2)
        stride_attn_mask_kv_seqlen = attn_mask.stride(3)
    else:
        HAS_ATTN_MASK = False
        stride_attn_mask_batch = 1
        stride_attn_mask_head = 1
        stride_attn_mask_q_seqlen = 1
        stride_attn_mask_kv_seqlen = 1

    M = torch.empty(
        (query.shape[0], query.shape[1], query.shape[2]),
        device=query.device,
        dtype=torch.float32,
    )

    with torch_device_fn.device(query.device):
        _attn_fwd_d256[grid](
            query,
            key,
            value,
            attn_mask,
            sm_scale,
            M,
            o,  #
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),  #
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),  #
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),  #
            stride_attn_mask_batch,
            stride_attn_mask_head,
            stride_attn_mask_q_seqlen,
            stride_attn_mask_kv_seqlen,  #
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),  #
            query.shape[0],
            q_head_num,
            kv_head_num,  #
            q_head_num // kv_head_num,  # group_head
            query.shape[2],  #
            key.shape[2],  #
            HEAD_DIM_K,  #
            STAGE=stage,  #
            HAS_ATTN_MASK=HAS_ATTN_MASK,  #
            # Removing tail-fill buffers in the guarded aligned path makes
            # this larger tile fit UB without any accumulator downcast.
            BLOCK_M=64,
            BLOCK_N=256 if query.shape[2] % 256 == 0 else 128,
            PRE_LOAD_V=False,
            num_warps=4,
            num_stages=1,
            sync_solver=False,
        )
    return o, M


@libentry()
@triton.jit
def _attn_bwd_d256_materialize(
    Q,
    K,
    V,
    DO,
    M,
    Delta,
    P,
    DS,
    H: tl.constexpr,
    KV_HEADS: tl.constexpr,
    S: tl.constexpr,
    sm_scale: tl.constexpr,
    head_id,
    BLOCK: tl.constexpr = 128,
    TRANSPOSED_SCORES: tl.constexpr = False,
):
    # At 8K, key-major scores give dK/dV contiguous query spans. Shorter
    # cached sequences retain query-major scores: transposing them regressed
    # the measured 4K workload. Consumers skip the unwritten causal region.
    row = tl.program_id(0)
    head_id = head_id + tl.program_id(1)
    kv_head = (head_id // H) * KV_HEADS + (head_id % H) // (H // KV_HEADS)
    q_offset = head_id.to(tl.int64) * S * 256
    kv_offset = kv_head.to(tl.int64) * S * 256
    m = row * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, 256)
    q = tl.load(Q + q_offset + m[:, None] * 256 + d[None, :])
    grad = tl.load(DO + q_offset + m[:, None] * 256 + d[None, :])
    lse = tl.load(M + head_id * S + m)
    delta = tl.load(Delta + head_id * S + m)
    for col in range(row + 1):
        n = col * BLOCK + tl.arange(0, BLOCK)
        if TRANSPOSED_SCORES:
            key = tl.load(K + kv_offset + n[:, None] * 256 + d[None, :])
            scores = tl.dot(key, tl.trans(q)) * sm_scale * 1.44269504
            prob = tl.exp2(scores - lse[None, :])
            prob = tl.where(m[None, :] >= n[:, None], prob, 0.0)
            value = tl.load(V + kv_offset + n[:, None] * 256 + d[None, :])
            dp = tl.dot(value, tl.trans(grad))
            ds = prob * (dp - delta[None, :])
            offsets = n[:, None] * S + m[None, :]
        else:
            key = tl.load(K + kv_offset + n[None, :] * 256 + d[:, None])
            scores = tl.dot(q, key) * sm_scale * 1.44269504
            prob = tl.exp2(scores - lse[:, None])
            prob = tl.where(m[:, None] >= n[None, :], prob, 0.0)
            value = tl.load(V + kv_offset + n[None, :] * 256 + d[:, None])
            dp = tl.dot(grad, value)
            ds = prob * (dp - delta[:, None])
            offsets = m[:, None] * S + n[None, :]
        # Match the existing dot-input casts; all score and dS math is FP32.
        tl.store(P + offsets, prob)
        tl.store(DS + offsets, ds)


@libentry()
@triton.jit
def _attn_bwd_d256_cached_grad(
    A,
    X,
    Y,
    H: tl.constexpr,
    KV_HEADS: tl.constexpr,
    S: tl.constexpr,
    sm_scale: tl.constexpr,
    COMPUTE_DQ: tl.constexpr,
    head_id,
    BLOCK: tl.constexpr = 128,
    TRANSPOSED_SCORES: tl.constexpr = False,
):
    row = tl.program_id(0)
    head_id = head_id + tl.program_id(1)
    m = row * BLOCK + tl.arange(0, BLOCK)
    r = tl.arange(0, BLOCK)
    d = tl.arange(0, 256)
    acc = tl.zeros((BLOCK, 256), tl.float32)
    if COMPUTE_DQ:
        x_head = (head_id // H) * KV_HEADS + (head_id % H) // (H // KV_HEADS)
        begin, end = 0, (row + 1) * BLOCK
    else:
        x_head = head_id
        begin, end = row * BLOCK, S
    x_offset = x_head.to(tl.int64) * S * 256
    for start in range(begin, end, BLOCK):
        k = start + r
        # Stored-score transposition swaps the two gradient orientations.
        if COMPUTE_DQ != TRANSPOSED_SCORES:
            a = tl.load(A + m[:, None] * S + k[None, :])
        else:
            a = tl.load(A + k[None, :] * S + m[:, None])
        x = tl.load(X + x_offset + k[:, None] * 256 + d[None, :])
        acc += tl.dot(a, x)
    out_offset = head_id.to(tl.int64) * S * 256
    tl.store(Y + out_offset + m[:, None] * 256 + d[None, :], acc * sm_scale)


@libentry()
@triton.jit
def _attn_bwd_d256_stripe_materialize(
    Q,
    K,
    V,
    DO,
    M,
    Delta,
    P,
    DS,
    H: tl.constexpr,
    KH: tl.constexpr,
    S: tl.constexpr,
    sm_scale: tl.constexpr,
    head_id,
    q_start,
    BLOCK: tl.constexpr = 128,
    SEGMENTS: tl.constexpr = 1,
):
    row = tl.program_id(0)
    segment = tl.program_id(1)
    head_id = head_id + tl.program_id(2)
    q_start = q_start + tl.program_id(2)
    m = q_start + row * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, 256)
    kv_head = (head_id // H) * KH + (head_id % H) // (H // KH)
    q_off = head_id.to(tl.int64) * S * 256
    k_off = kv_head.to(tl.int64) * S * 256
    # Reuse each row's Q/dO/LSE/Delta across its assigned key tiles. Segments
    # own distinct column residues, so they never overlap in P or dS.
    q = tl.load(Q + q_off + m[:, None] * 256 + d[None, :])
    grad = tl.load(DO + q_off + m[:, None] * 256 + d[None, :])
    lse = tl.load(M + head_id * S + m)
    delta = tl.load(Delta + head_id * S + m)
    for col in range(segment, q_start // BLOCK + row + 1, SEGMENTS):
        n = col * BLOCK + tl.arange(0, BLOCK)
        k = tl.load(K + k_off + n[None, :] * 256 + d[:, None])
        scores = tl.dot(q, k) * sm_scale * 1.44269504
        p = tl.exp2(scores - lse[:, None])
        p = tl.where(m[:, None] >= n[None, :], p, 0.0)
        v = tl.load(V + k_off + n[None, :] * 256 + d[:, None])
        dp = tl.dot(grad, v)
        ds = p * (dp - delta[:, None])
        # Store each score tile contiguously: both gradient orientations
        # consume it without the full-sequence stride between tile rows.
        off = (
            (row * (S // BLOCK) + col) * BLOCK * BLOCK
            + tl.arange(0, BLOCK)[:, None] * BLOCK
            + tl.arange(0, BLOCK)[None, :]
        )
        tl.store(P + off, p)
        tl.store(DS + off, ds)


@libentry()
@triton.jit
def _attn_bwd_d256_stripe_dq(
    DS,
    K,
    DQ,
    H: tl.constexpr,
    KH: tl.constexpr,
    S: tl.constexpr,
    sm_scale: tl.constexpr,
    head_id,
    q_start,
    BLOCK: tl.constexpr = 128,
    FEATURE: tl.constexpr = 128,
):
    row = tl.program_id(0)
    head_id = head_id + tl.program_id(2)
    q_start = q_start + tl.program_id(2)
    lm = row * BLOCK + tl.arange(0, BLOCK)
    m = q_start + lm
    d = tl.program_id(1) * FEATURE + tl.arange(0, FEATURE)
    r = tl.arange(0, BLOCK)
    kv_head = (head_id // H) * KH + (head_id % H) // (H // KH)
    k_off = kv_head.to(tl.int64) * S * 256
    acc = tl.zeros((BLOCK, FEATURE), tl.float32)
    for start in range(0, q_start + (row + 1) * BLOCK, BLOCK):
        n = start + r
        tile = (row * (S // BLOCK) + start // BLOCK) * BLOCK * BLOCK
        a = tl.load(DS + tile + r[:, None] * BLOCK + r[None, :])
        k = tl.load(K + k_off + n[:, None] * 256 + d[None, :])
        acc += tl.dot(a, k)
    out_off = head_id.to(tl.int64) * S * 256
    tl.store(DQ + out_off + m[:, None] * 256 + d[None, :], acc * sm_scale)


@libentry()
@triton.jit
def _attn_bwd_d256_stripe_dkv(
    A,
    X,
    Partial,
    Out,
    S: tl.constexpr,
    sm_scale: tl.constexpr,
    head_id,
    q_start,
    SPAN: tl.constexpr,
    FINAL: tl.constexpr,
    BLOCK: tl.constexpr = 128,
    FEATURE: tl.constexpr = 128,
):
    row = tl.program_id(0)
    head_id = head_id + tl.program_id(2)
    q_start = q_start + tl.program_id(2)
    n = row * BLOCK + tl.arange(0, BLOCK)
    d = tl.program_id(1) * FEATURE + tl.arange(0, FEATURE)
    r = tl.arange(0, BLOCK)
    x_off = head_id.to(tl.int64) * S * 256
    # A key tile is initialized by the first stripe intersecting its diagonal.
    # Earlier stripes never wrote the future key tiles in Partial.
    # Keep the dot accumulator in Cube storage throughout this stripe.
    # Loading the previous stripe into it forces a GM/Vector update per dot.
    # Combine FP32 stripe sums after the dot loop; cast only at final output.
    # This groups FP32 additions by stripe instead of initializing each dot
    # loop with the prior sum, so accuracy is checked against the fused path.
    acc = tl.zeros((BLOCK, FEATURE), tl.float32)
    begin = tl.maximum(q_start, row * BLOCK)
    for start in range(begin, q_start + SPAN, BLOCK):
        m = start + r
        tile = (((start - q_start) // BLOCK) * (S // BLOCK) + row) * BLOCK * BLOCK
        a = tl.load(A + tile + r[None, :] * BLOCK + r[:, None])
        x = tl.load(X + x_off + m[:, None] * 256 + d[None, :])
        acc += tl.dot(a, x)
    prior = tl.load(
        Partial + n[:, None] * 256 + d[None, :],
        mask=row * BLOCK < q_start,
        other=0.0,
    )
    acc += prior
    if FINAL:
        tl.store(Out + x_off + n[:, None] * 256 + d[None, :], acc * sm_scale)
    else:
        tl.store(Partial + n[:, None] * 256 + d[None, :], acc)


def _aligned_d256_backward_striped(
    do,
    query,
    key,
    value,
    o,
    M,
    *,
    scale=None,
    stripe_rows=3072,
    feature_block=256,
    materialize_segments=None,
):
    from .sum import sum_dim

    b, h, s, d = query.shape
    kh = key.shape[1]
    assert d == 256 and s % 128 == 0
    assert stripe_rows > 0 and stripe_rows % 128 == 0
    assert feature_block in (64, 128, 256)
    assert materialize_segments is None or (
        isinstance(materialize_segments, int) and materialize_segments > 0
    )
    factor = 1.0 / 16 if scale is None else scale
    stripe_rows = min(stripe_rows, s)
    delta = torch.empty_like(M)
    dq = torch.empty_like(query)
    dk = torch.empty_like(query)
    dv = torch.empty_like(query)
    # At 32K, P/DS occupy 384 MiB and FP32 partials another 64 MiB.
    # 3072 rows give 24 dQ tiles with all 256 features per Cube program.
    # Keep the same tile storage when a final stripe is shorter.
    probabilities = torch.empty(
        (stripe_rows // 128, s // 128, 128, 128), device=query.device, dtype=query.dtype
    )
    ds = torch.empty_like(probabilities)
    partial_k = torch.empty((s, 256), device=query.device, dtype=torch.float32)
    partial_v = torch.empty_like(partial_k)
    _attn_bwd_d256_preprocess[(s // 32, b * h)](
        o,
        do,
        delta,
        b,
        h,
        s,
        BLOCK_M=32,
        D_HEAD=256,
        num_warps=4,
        num_stages=1,
    )
    # All consumers run in the current stream before a stripe/head reuses P/DS.
    # Gradient partials remain FP32 across stripes; there are no atomic updates.
    for head_id in range(b * h):
        for q_start in range(0, s, stripe_rows):
            span = min(stripe_rows, s - q_start)
            end = q_start + span
            segments = (
                max(1, triton.cdiv(32, span // 128))
                if materialize_segments is None
                else materialize_segments
            )
            _attn_bwd_d256_stripe_materialize[(span // 128, segments)](
                query,
                key,
                value,
                do,
                M,
                delta,
                probabilities,
                ds,
                h,
                kh,
                s,
                factor,
                head_id,
                q_start,
                SEGMENTS=segments,
                num_warps=4,
                num_stages=2,
                sync_solver=False,
            )
            _attn_bwd_d256_stripe_dq[(span // 128, 256 // feature_block)](
                ds,
                key,
                dq,
                h,
                kh,
                s,
                factor,
                head_id,
                q_start,
                FEATURE=feature_block,
                num_warps=4,
                num_stages=2,
                sync_solver=False,
            )
            for a, x, partial, out, multiplier in (
                (ds, query, partial_k, dk, factor),
                (probabilities, do, partial_v, dv, 1.0),
            ):
                _attn_bwd_d256_stripe_dkv[(end // 128, 256 // feature_block)](
                    a,
                    x,
                    partial,
                    out,
                    s,
                    multiplier,
                    head_id,
                    q_start,
                    span,
                    end == s,
                    FEATURE=feature_block,
                    num_warps=4,
                    num_stages=2,
                    sync_solver=False,
                )
    if h != kh:
        shape = (b, kh, h // kh, s, 256)
        dk = sum_dim(dk.view(shape), [2])
        dv = sum_dim(dv.view(shape), [2])
    return dq, dk, dv


def _aligned_d256_backward(
    do,
    query,
    key,
    value,
    o,
    M,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    assert dropout_p == 0.0, "Currenty only support dropout_p=0.0"

    if scale is None:
        sm_scale = 1.0 / (HEAD_DIM_K**0.5)
    else:
        sm_scale = scale

    assert do.is_contiguous()
    assert (
        query.is_contiguous()
        and key.is_contiguous()
        and value.is_contiguous()
        and o.is_contiguous()
    )
    assert query.stride() == o.stride() == do.stride()
    assert key.stride() == value.stride()

    BLOCK_DMODEL = HEAD_DIM_K
    BATCH, Q_HEAD, Q_CTX = query.shape[:3]
    _, KV_HEAD, KV_CTX = key.shape[:3]
    group_head = Q_HEAD // KV_HEAD

    # Keep the bounded-cache route limited to the measured 32K workload.
    if Q_CTX == 32768:
        return _aligned_d256_backward_striped(do, query, key, value, o, M, scale=scale)

    # Separate dK and dV so their FP32 accumulators are not simultaneously
    # live in L0C. The dK/dV query traversal is aligned down to M1; the causal
    # mask removes the extra prefix when M1 > N1.
    loop_tile = 256 if Q_CTX % 256 == 0 else 128
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = loop_tile, 64, 64, loop_tile

    # Scale QK in FP32 inside the kernels. Pre-scaling K in BF16/FP16
    # adds rounding before probability reconstruction and gradient accumulation.
    arg_k = key
    PRE_BLOCK = 32
    pre_grid = (triton.cdiv(Q_CTX, PRE_BLOCK), BATCH * Q_HEAD)

    delta = torch.empty_like(M)

    # NOTE that dk & dv always have the same number of heads as q
    dq = torch.empty_like(query).contiguous()
    dk = torch.empty(
        (BATCH, Q_HEAD, KV_CTX, HEAD_DIM_K),
        device=key.device,
        dtype=key.dtype,
        memory_format=torch.contiguous_format,
    )
    dv = torch.empty(
        (BATCH, Q_HEAD, KV_CTX, HEAD_DIM_V),
        device=value.device,
        dtype=value.dtype,
        memory_format=torch.contiguous_format,
    )

    _attn_bwd_d256_preprocess[pre_grid](
        o,
        do,  #
        delta,  #
        BATCH,
        Q_HEAD,
        Q_CTX,  #
        BLOCK_M=PRE_BLOCK,
        D_HEAD=BLOCK_DMODEL,  #
        num_warps=4,
        num_stages=1,
    )

    # On long, many-head workloads, one-head score reuse avoids recomputing
    # QK/dP for each gradient. Bound the two FP16/BF16 buffers to 256 MiB.
    # This range has lower measured total peak memory than the fused path.
    use_score_cache = (
        4096 <= Q_CTX <= 8192 and Q_CTX % 256 == 0 and BATCH * Q_HEAD >= 16
    )
    if use_score_cache:
        probabilities = torch.empty(
            (Q_CTX, KV_CTX), device=query.device, dtype=query.dtype
        )
        ds = torch.empty_like(probabilities)
        grid = (Q_CTX // 128,)
        for head_id in range(BATCH * Q_HEAD):
            _attn_bwd_d256_materialize[grid](
                query,
                key,
                value,
                do,
                M,
                delta,
                probabilities,
                ds,
                Q_HEAD,
                KV_HEAD,
                Q_CTX,
                sm_scale,
                head_id,
                TRANSPOSED_SCORES=Q_CTX == 8192,
                num_warps=4,
                num_stages=1,
                sync_solver=False,
            )
            for scores, operand, gradient, compute_dq, factor in (
                (ds, key, dq, True, sm_scale),
                (ds, query, dk, False, sm_scale),
                (probabilities, do, dv, False, 1.0),
            ):
                _attn_bwd_d256_cached_grad[grid](
                    scores,
                    operand,
                    gradient,
                    Q_HEAD,
                    KV_HEAD,
                    Q_CTX,
                    factor,
                    compute_dq,
                    head_id,
                    TRANSPOSED_SCORES=Q_CTX == 8192,
                    num_warps=4,
                    num_stages=2,
                    sync_solver=False,
                )
    else:
        for compute_dq, grad_kv in ((False, 1), (False, 2), (True, 0)):
            seq_len, block = (Q_CTX, BLOCK_M2) if compute_dq else (KV_CTX, BLOCK_N1)
            grid = (
                (triton.cdiv(seq_len, block), 1, BATCH * Q_HEAD)
                if compute_dq
                else (BATCH * Q_HEAD, 1, triton.cdiv(seq_len, block))
            )
            _attn_bwd_d256[grid](
                query,
                arg_k,
                value,
                sm_scale,
                do,
                dq,
                dk,
                dv,  #
                M,
                delta,  #
                query.stride(0),
                query.stride(1),
                query.stride(2),
                query.stride(3),  #
                key.stride(0),
                key.stride(1),  #
                Q_HEAD,
                Q_CTX,  #
                KV_CTX,  #
                KV_HEAD,  #
                GROUP_HEAD=group_head,  #
                BLOCK_M1=BLOCK_M1,
                BLOCK_N1=BLOCK_N1,
                BLOCK_M2=BLOCK_M2,
                BLOCK_N2=BLOCK_N2,
                BLOCK_DMODEL=BLOCK_DMODEL,  #
                num_warps=4,
                num_stages=1,
                # Keep graph sync solving for dV's peeled loops. Use legacy sync
                # insertion only for the validated single-loop dK/dQ kernels.
                sync_solver=grad_kv == 2,
                COMPUTE_DQ=compute_dq,
                GRAD_KV=grad_kv,
                IS_CAUSAL=is_causal,
            )

    if group_head > 1:
        dk = dk.reshape(BATCH, Q_HEAD // group_head, group_head, KV_CTX, HEAD_DIM_K)
        dv = dv.reshape(BATCH, Q_HEAD // group_head, group_head, KV_CTX, HEAD_DIM_V)
        dk = dk.sum(dim=2)
        dv = dv.sum(dim=2)

    return dq, dk, dv


def scaled_dot_product_attention_forward(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    # Aligned D256 fast path. Keep irregular lengths, other dtypes, masks and
    # noncausal attention on the implementation below.
    if _use_aligned_d256(
        query, key, value, attn_mask, dropout_p, is_causal, enable_gqa
    ):
        return _aligned_d256_forward(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    logger.debug("GEMS_ASCEND SCALED_DOT_PRODUCT_ATTENTION_FORWARD")
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert 16 <= HEAD_DIM_K <= 256
    assert dropout_p == 0.0, "Currenty only support dropout_p=0.0"

    o = torch.empty_like(query, dtype=value.dtype)

    stage = 3 if is_causal else 1

    if scale is None:
        sm_scale = 1.0 / (HEAD_DIM_K**0.5)
    else:
        sm_scale = scale

    # Keep scaling and tensor strides based on the real head dimension. Only
    # the kernel tile is padded; masked loads supply zeros for the extra lanes.
    HEAD_DIM_ACTUAL = HEAD_DIM_K
    HEAD_DIM_K = triton.next_power_of_2(HEAD_DIM_K)

    q_head_num = query.shape[1]
    kv_head_num = key.shape[1]
    assert enable_gqa or q_head_num == kv_head_num, (
        f"q_head_num {q_head_num} != kv_head_num {kv_head_num}, "
        "enable_gqa must be True to support different head numbers."
    )

    grid = lambda args: (
        triton.cdiv(query.shape[2], args["BLOCK_M"]),
        query.shape[0] * query.shape[1],
        1,
    )

    if attn_mask is not None:
        HAS_ATTN_MASK = True
        if attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.to(query.dtype) * -1.0e6
        # The kernel indexes [B, H_q, S_q, S_k], so singleton and omitted
        # mask dimensions must have zero stride after broadcasting.
        (
            stride_attn_mask_batch,
            stride_attn_mask_head,
            stride_attn_mask_q_seqlen,
            stride_attn_mask_kv_seqlen,
        ) = broadcasted_stride(
            attn_mask.shape,
            attn_mask.stride(),
            (query.shape[0], q_head_num, query.shape[2], key.shape[2]),
        )
    else:
        HAS_ATTN_MASK = False
        stride_attn_mask_batch = 1
        stride_attn_mask_head = 1
        stride_attn_mask_q_seqlen = 1
        stride_attn_mask_kv_seqlen = 1

    M = torch.empty(
        (query.shape[0], query.shape[1], query.shape[2]),
        device=query.device,
        dtype=torch.float32,
    )

    with torch_device_fn.device(query.device):
        _attn_fwd[grid](
            query,
            key,
            value,
            attn_mask,
            sm_scale,
            M,
            o,  #
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),  #
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),  #
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),  #
            stride_attn_mask_batch,
            stride_attn_mask_head,
            stride_attn_mask_q_seqlen,
            stride_attn_mask_kv_seqlen,  #
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),  #
            query.shape[0],
            q_head_num,
            kv_head_num,  #
            q_head_num // kv_head_num,  # group_head
            query.shape[2],  #
            key.shape[2],  #
            HEAD_DIM_K,  #
            HEAD_DIM_ACTUAL=HEAD_DIM_ACTUAL,
            STAGE=stage,  #
            HAS_ATTN_MASK=HAS_ATTN_MASK,  #
            # D=256 benefits from longer KV tiles with row-major K loads.
            # Keep M=32: M=64 overflows UB even without multibuffering.
            BLOCK_M=32,
            BLOCK_N=64 if HEAD_DIM_K == 256 else min(32, HEAD_DIM_K),
            PRE_LOAD_V=False,
            num_warps=4,
            num_stages=1,
            sync_solver=True,
        )
    return o, M


def scaled_dot_product_attention_backward(
    do,
    query,
    key,
    value,
    o,
    M,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    # Aligned D256 fast path. The fused kernels also require contiguous
    # gradient, output and softmax-statistics tensors.
    if _use_aligned_d256(
        query, key, value, attn_mask, dropout_p, is_causal, enable_gqa
    ) and all(x.is_contiguous() for x in (do, o, M)):
        return _aligned_d256_backward(
            do,
            query,
            key,
            value,
            o,
            M,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    logger.debug("GEMS_ASCEND SCALED_DOT_PRODUCT_ATTENTION_BACKWARD")
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    assert dropout_p == 0.0, "Currenty only support dropout_p=0.0"

    if scale is None:
        sm_scale = 1.0 / (HEAD_DIM_K**0.5)
    else:
        sm_scale = scale

    assert do.is_contiguous()
    assert (
        query.is_contiguous()
        and key.is_contiguous()
        and value.is_contiguous()
        and o.is_contiguous()
    )
    assert query.stride() == o.stride() == do.stride()
    assert key.stride() == value.stride()

    BLOCK_DMODEL = HEAD_DIM_K
    BATCH, Q_HEAD, Q_CTX = query.shape[:3]
    _, KV_HEAD, KV_CTX = key.shape[:3]
    group_head = Q_HEAD // KV_HEAD

    # Bound live accumulators and temporary buffers for the 192 KiB UB.
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 32, 32, 32
    # dQ has one gradient accumulator and fits a larger query tile. Keep
    # dK/dV at N=32; N=64 exceeds the accumulator buffer for D=256.
    if HEAD_DIM_K == 256:
        BLOCK_M2 = 64

    RCP_LN2 = 1.0 / math.log(2)

    arg_k = key * (sm_scale * RCP_LN2)
    PRE_BLOCK = 32
    pre_grid = (triton.cdiv(Q_CTX, PRE_BLOCK), BATCH * Q_HEAD)

    delta = torch.empty_like(M)

    # NOTE that dk & dv always have the same number of heads as q
    dq = torch.empty_like(query).contiguous()
    dk = torch.empty(
        (BATCH, Q_HEAD, KV_CTX, HEAD_DIM_K),
        device=key.device,
        dtype=key.dtype,
        memory_format=torch.contiguous_format,
    )
    dv = torch.empty(
        (BATCH, Q_HEAD, KV_CTX, HEAD_DIM_V),
        device=value.device,
        dtype=value.dtype,
        memory_format=torch.contiguous_format,
    )

    _attn_bwd_preprocess[pre_grid](
        o,
        do,  #
        delta,  #
        BATCH,
        Q_HEAD,
        Q_CTX,  #
        BLOCK_M=PRE_BLOCK,
        D_HEAD=BLOCK_DMODEL,  #
        num_warps=4,
        num_stages=1,
    )

    for compute_dq in (False, True):
        seq_len, block = (Q_CTX, BLOCK_M2) if compute_dq else (KV_CTX, BLOCK_N1)
        grid = (triton.cdiv(seq_len, block), 1, BATCH * Q_HEAD)
        _attn_bwd[grid](
            query,
            arg_k,
            value,
            sm_scale,
            do,
            dq,
            dk,
            dv,  #
            M,
            delta,  #
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),  #
            key.stride(0),
            key.stride(1),  #
            Q_HEAD,
            Q_CTX,  #
            KV_CTX,  #
            KV_HEAD,  #
            GROUP_HEAD=group_head,  #
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,
            BLOCK_DMODEL=BLOCK_DMODEL,  #
            num_warps=4,
            num_stages=1,
            sync_solver=True,
            COMPUTE_DQ=compute_dq,
            IS_CAUSAL=is_causal,
        )

    if group_head > 1:
        dk = dk.reshape(BATCH, Q_HEAD // group_head, group_head, KV_CTX, HEAD_DIM_K)
        dv = dv.reshape(BATCH, Q_HEAD // group_head, group_head, KV_CTX, HEAD_DIM_V)
        dk = dk.sum(dim=2)
        dv = dv.sum(dim=2)

    return dq, dk, dv


class ScaleDotProductAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        sm_scale = scale if scale is not None else 1.0 / (key.shape[-1] ** 0.5)
        o, M = scaled_dot_product_attention_forward(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            sm_scale,
            enable_gqa,
        )

        ctx.save_for_backward(query, key, value, o, M)
        ctx.sm_scale = sm_scale
        ctx.causal = is_causal
        ctx.enable_gqa = enable_gqa
        return o

    @staticmethod
    def backward(ctx, do):
        query, key, value, o, M = ctx.saved_tensors
        is_causal = ctx.causal
        enable_gqa = ctx.enable_gqa
        sm_scale = ctx.sm_scale
        dq, dk, dv = scaled_dot_product_attention_backward(
            do,
            query,
            key,
            value,
            o,
            M,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=sm_scale,
            enable_gqa=enable_gqa,
        )
        return dq, dk, dv, None, None, None, None, None


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    return ScaleDotProductAttention.apply(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa,
    )


def flash_attention_forward(
    query,
    key,
    value,
    cumulative_sequence_length_q,
    cumulative_sequence_length_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    return_debug_mask,
    *,
    scale=None,
    softcap=0.0,
    window_size_left=None,
    window_size_right=None,
    seqused_k=None,
    alibi_slopes=None,
    disable_splitkv=False,
):
    logger.debug("GEMS_ASCEND FLASH_ATTENTION_FORWARD")
    assert (cumulative_sequence_length_q is None) == (
        cumulative_sequence_length_k is None
    ), "Both cumulative lengths must be supplied"

    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    original_head_dim = HEAD_DIM_K
    supported_head_dims = (16, 32, 64, 96, 128, 192, 256)
    if HEAD_DIM_K not in supported_head_dims:
        padded_head_dim = None
        for d in supported_head_dims:
            if d >= HEAD_DIM_K:
                padded_head_dim = d
                break
        assert (
            padded_head_dim is not None
        ), f"Unsupported head dim {HEAD_DIM_K}, max supported is {supported_head_dims[-1]}"
        pad = padded_head_dim - HEAD_DIM_K
        query = F.pad(query, (0, pad))
        key = F.pad(key, (0, pad))
        value = F.pad(value, (0, pad))
        HEAD_DIM_K = padded_head_dim

    softmax_scale = scale or 1.0 / (original_head_dim**0.5)
    if window_size_left is not None:
        non_null_window_left = window_size_left
    else:
        non_null_window_left = -1
    if window_size_right is not None:
        non_null_window_right = window_size_right
    else:
        non_null_window_right = -1

    out = torch.empty_like(query)
    if cumulative_sequence_length_q is not None:
        out, q, k, v, lse, philox_seed, philox_offset, p = mha_varlan_fwd(
            query,
            key,
            value,
            out,
            cumulative_sequence_length_q,
            cumulative_sequence_length_k,
            seqused_k,
            None,
            None,  # block_table
            alibi_slopes,
            max_q,
            max_k,
            dropout_p,
            softmax_scale,
            False,
            is_causal,
            non_null_window_left,
            non_null_window_right,
            softcap,
            return_debug_mask and dropout_p > 0,
            None,
        )
    else:
        out, q, k, v, lse, philox_seed, philox_offset, p = mha_fwd(
            query,
            key,
            value,
            out,
            alibi_slopes,
            dropout_p,
            softmax_scale,
            is_causal,
            non_null_window_left,
            non_null_window_right,
            softcap,
            return_debug_mask,
            disable_splitkv=disable_splitkv,
        )

    if HEAD_DIM_K != original_head_dim:
        out = out[..., :original_head_dim]
    return (out, lse, philox_seed, philox_offset, p)


# Adapted from https://github.com/vllm-project/flash-attention/blob/main/vllm_flash_attn/flash_attn_interface.py
def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,  # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # Dummy FA3 arguments
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k=None,
    fa_version: int = 2,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    if fa_version != 2:
        raise RuntimeError("Only FA2 is implemented.")
    if num_splits > 0:
        raise RuntimeError("num_splits > 0 is not implemented in GEMS.")
    if use_c_extension:
        logger.debug("GEMS_ASCEND FLASH_ATTN_VARLEN_FUNC")
        with torch_device_fn.device(q.device):
            out_cpp, softmax_lse = torch.ops.flag_gems.flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q,
                cu_seqlens_q,
                max_seqlen_k,
                cu_seqlens_k,
                seqused_k,
                q_v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                softcap,
                alibi_slopes,
                deterministic,
                return_attn_probs,
                block_table,
                return_softmax_lse,
                out,
                scheduler_metadata,
                q_descale,
                k_descale,
                v_descale,
                s_aux,
                num_splits,
                cp_world_size,
                cp_rank,
                cp_tot_seqused_k,
                fa_version,
            )
        return (out_cpp, softmax_lse) if return_softmax_lse else out_cpp
    else:
        logger.debug("GEMS_ASCEND FLASH_ATTN_VARLEN_FUNC")
        assert (
            cu_seqlens_k is not None or seqused_k is not None
        ), "cu_seqlens_k or seqused_k must be provided"
        assert (
            cu_seqlens_k is None or seqused_k is None
        ), "cu_seqlens_k and seqused_k cannot be provided at the same time"
        assert (
            block_table is None or seqused_k is not None
        ), "seqused_k must be provided if block_table is provided"
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        # custom op does not support non-tuple input
        if window_size is None:
            real_window_size = (-1, -1)
        else:
            assert len(window_size) == 2
            real_window_size = (window_size[0], window_size[1])
        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
        dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)
        max_seqlen_q = (
            max_seqlen_q.item() if hasattr(max_seqlen_q, "item") else max_seqlen_q
        )
        max_seqlen_k = (
            max_seqlen_k.item() if hasattr(max_seqlen_k, "item") else max_seqlen_k
        )
        out, q, k, v, softmax_lse, *_ = mha_varlan_fwd(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            # cu_seqlens_k not used since we use seqused_k, but flash_api.cpp
            # still wants it so we pass all zeros
            dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
            seqused_k,
            None,
            block_table,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            real_window_size[0],
            real_window_size[1],
            softcap,
            return_softmax_lse and dropout_p > 0,
            None,
        )

    return (out, softmax_lse) if return_softmax_lse else out


def flash_attn_varlen_opt_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,  # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    lse=None,
    # Dummy FA3 arguments
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k=None,
    fa_version: int = 2,
):
    """Ascend variable-length attention with optional preallocated output and LSE."""
    if fa_version != 2:
        raise RuntimeError("Only FA2 is implemented.")
    if num_splits > 0:
        raise RuntimeError("num_splits > 0 is not implemented in GEMS.")
    if use_c_extension:
        logger.debug("GEMS FLASH_ATTN_VARLEN_FUNC(C EXTENSION)")
        with torch_device_fn.device(q.device):
            out_cpp, softmax_lse = torch.ops.flag_gems.flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q,
                cu_seqlens_q,
                max_seqlen_k,
                cu_seqlens_k,
                seqused_k,
                q_v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                softcap,
                alibi_slopes,
                deterministic,
                return_attn_probs,
                block_table,
                return_softmax_lse,
                out,
                scheduler_metadata,
                q_descale,
                k_descale,
                v_descale,
                s_aux,
                num_splits,
                cp_world_size,
                cp_rank,
                cp_tot_seqused_k,
                fa_version,
            )
        return (out_cpp, softmax_lse) if return_softmax_lse else out_cpp
    else:
        logger.debug("GEMS_ASCEND FLASH_ATTN_VARLEN_OPT_FUNC")
        assert (
            cu_seqlens_k is not None or seqused_k is not None
        ), "cu_seqlens_k or seqused_k must be provided"
        assert (
            cu_seqlens_k is None or seqused_k is None
        ), "cu_seqlens_k and seqused_k cannot be provided at the same time"
        assert (
            block_table is None or seqused_k is not None
        ), "seqused_k must be provided if block_table is provided"
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        # custom op does not support non-tuple input
        if window_size is None:
            real_window_size = (-1, -1)
        else:
            assert len(window_size) == 2
            real_window_size = (window_size[0], window_size[1])
        max_seqlen_q = (
            max_seqlen_q.item() if hasattr(max_seqlen_q, "item") else max_seqlen_q
        )
        max_seqlen_k = (
            max_seqlen_k.item() if hasattr(max_seqlen_k, "item") else max_seqlen_k
        )
        out, q, k, v, softmax_lse, *_ = mha_varlan_fwd_opt(
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            # Paged KV uses seqused_k; cumulative K lengths are unused there.
            cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k,
            seqused_k,
            None,
            block_table,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            real_window_size[0],
            real_window_size[1],
            softcap,
            return_softmax_lse and dropout_p > 0,
            None,
        )

    return (out, softmax_lse) if return_softmax_lse else out


def _flash_attention_forward(
    query,
    key,
    value,
    cumulative_sequence_length_q,
    cumulative_sequence_length_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    return_debug_mask,
    *,
    scale=None,
    window_size_left=None,
    window_size_right=None,
    seqused_k=None,
    alibi_slopes=None,
):
    """Run the canonical ATen entry through the existing
    FlashAttention implementation.
    """
    logger.debug("GEMS_ASCEND _FLASH_ATTENTION_FORWARD")
    return flash_attention_forward(
        query,
        key,
        value,
        cumulative_sequence_length_q,
        cumulative_sequence_length_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        seqused_k=seqused_k,
        alibi_slopes=alibi_slopes,
    )
