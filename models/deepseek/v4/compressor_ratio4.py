# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 KV Compressor (decode incremental, ratio=4 overlap).

State buffer layout (STATE_LEN = 2 * ratio = 8 slots, OUT_DIM = 2 * HEAD_DIM):
    slots 0..ratio-1  hold the previous (overlapping) window, valid in cols [0, HEAD_DIM)
    slots ratio..2*ratio-1 hold the current window, valid in cols [HEAD_DIM, OUT_DIM)
Softmax+pool tree-reduces across all 8 slots; the back half rolls forward into the
front half after compression so the next step starts with the fresh overlap window.

norm_w is BF16 here to mirror the official checkpoint format (`compressor.norm_w`
is stored as BF16). It is upcast to FP32 inside RMSNorm for the multiply."""


import pypto.language as pl


B = 16
S = 1
EPS = 1e-6

COMPRESS_RATIO = 4
HEAD_DIM = 512
ROTATE = False

D = 4096  # flash:4096 pro:7168
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)

OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO

START_POS = 3
SHOULD_COMPRESS = COMPRESS_RATIO != 0 and ((START_POS + 1) % COMPRESS_RATIO) == 0
APE_ROW = START_POS % COMPRESS_RATIO if COMPRESS_RATIO != 0 else 0
SCATTER_SLOT = (COMPRESS_RATIO + APE_ROW) if OVERLAP else APE_ROW

HEAD_DIM_INV = 1.0 / HEAD_DIM
HALF_RD = ROPE_HEAD_DIM // 2

K_CHUNK = 512
OUT_CHUNK = 64
K_BLOCKS = D // K_CHUNK
OUT_BLOCKS = OUT_DIM // OUT_CHUNK

HEAD_CHUNK = 128
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK


@pl.jit.inline
def compressor(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
    out: pl.Tensor[[B, HEAD_DIM], pl.BF16],
):
    x_flat = pl.reshape(x, [B, D])
    # Flatten state to 2D so scattering into a specific slot is a single slice-assign.
    kv_state_flat = pl.reshape(kv_state, [B, STATE_LEN * OUT_DIM])
    score_state_flat = pl.reshape(score_state, [B, STATE_LEN * OUT_DIM])
    slot_off = SCATTER_SLOT * OUT_DIM

    kv_fp32 = pl.create_tensor([B, OUT_DIM], dtype=pl.FP32)
    score_fp32 = pl.create_tensor([B, OUT_DIM], dtype=pl.FP32)

    # Stage 1: kv/score projections + APE bias + state scatter, tiled along OUT_DIM.
    for ob in pl.parallel(0, OUT_BLOCKS, 1):
        oc0 = ob * OUT_CHUNK

        # Cube: kv = x @ wkv.T
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj"):
            a0 = x_flat[:, :K_CHUNK]
            b0 = wkv[oc0 : oc0 + OUT_CHUNK, :K_CHUNK]
            kv_acc = pl.matmul(a0, b0, out_dtype=pl.FP32, b_trans=True)
            for kb in pl.range(1, K_BLOCKS):
                a_kb = x_flat[:, kb * K_CHUNK : (kb + 1) * K_CHUNK]
                b_kb = wkv[oc0 : oc0 + OUT_CHUNK, kb * K_CHUNK : (kb + 1) * K_CHUNK]
                kv_acc = pl.matmul_acc(kv_acc, a_kb, b_kb, b_trans=True)
            kv_fp32[:, oc0 : oc0 + OUT_CHUNK] = kv_acc

        # Cube: score = x @ wgate.T
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="score_proj"):
            ag0 = x_flat[:, :K_CHUNK]
            bg0 = wgate[oc0 : oc0 + OUT_CHUNK, :K_CHUNK]
            sc_acc = pl.matmul(ag0, bg0, out_dtype=pl.FP32, b_trans=True)
            for kb in pl.range(1, K_BLOCKS):
                ag_kb = x_flat[:, kb * K_CHUNK : (kb + 1) * K_CHUNK]
                bg_kb = wgate[oc0 : oc0 + OUT_CHUNK, kb * K_CHUNK : (kb + 1) * K_CHUNK]
                sc_acc = pl.matmul_acc(sc_acc, ag_kb, bg_kb, b_trans=True)
            score_fp32[:, oc0 : oc0 + OUT_CHUNK] = sc_acc

        # Vector: score += ape[APE_ROW] (broadcast across batch via [B,1]-ones × [1,N])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ape_add"):
            ape_row = ape[APE_ROW : APE_ROW + 1, oc0 : oc0 + OUT_CHUNK]
            ones_b = pl.full([B, OUT_CHUNK], dtype=pl.FP32, value=1.0)
            ape_bcast = pl.col_expand_mul(ones_b, ape_row)
            sc = score_fp32[:, oc0 : oc0 + OUT_CHUNK]
            score_fp32[:, oc0 : oc0 + OUT_CHUNK] = pl.add(sc, ape_bcast)

        # Vector: scatter this step's kv/score into state slot SCATTER_SLOT.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter"):
            kv_state_flat[:, slot_off + oc0 : slot_off + oc0 + OUT_CHUNK] = kv_fp32[:, oc0 : oc0 + OUT_CHUNK]
            score_state_flat[:, slot_off + oc0 : slot_off + oc0 + OUT_CHUNK] = score_fp32[:, oc0 : oc0 + OUT_CHUNK]

    # Per-state-row view: rows are stacked as [B * STATE_LEN, OUT_DIM] so a slice on
    # the row dim addresses one (batch, slot) pair.
    kv_state_per_row = pl.reshape(kv_state_flat, [B * STATE_LEN, OUT_DIM])
    score_state_per_row = pl.reshape(score_state_flat, [B * STATE_LEN, OUT_DIM])

    # Stage 2 + 3: softmax+pool, then state shift, fused in a single CORE_GROUP
    # scope. Slots 0..3 live in cols [0, HEAD_DIM); slots 4..7 in cols
    # [HEAD_DIM, OUT_DIM). Tree reduction for max -> exp/sum -> weighted-sum.
    #
    # Two reasons everything must run as one kernel and not parallel-per-batch:
    #   1) `pooled` is a fresh create_tensor; the compiler signs it pl.Out and
    #      the orchestration emits add_output -> parallel AIV tasks would race
    #      on the buffer.
    #   2) If state_shift is its own pl.parallel(B) task, the scheduler runs
    #      shift iterations before the pool task completes (the pool's read of
    #      score_state_per_row doesn't fence shift's inout writes). Pool then
    #      sees post-shift state where slot 3 = old slot 7, mixing slot 3 +
    #      slot 7 in the softmax and producing wrong (but deterministic) output.
    # Fusing pool + shift per-batch in one kernel scope orders them strictly.
    pooled = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_pool"):
        for b_idx in pl.range(B):
            row_b = b_idx * STATE_LEN
            for hb in pl.range(HEAD_BLOCKS):
                h0 = hb * HEAD_CHUNK

                s0 = score_state_per_row[row_b + 0 : row_b + 1, h0 : h0 + HEAD_CHUNK]
                s1 = score_state_per_row[row_b + 1 : row_b + 2, h0 : h0 + HEAD_CHUNK]
                s2 = score_state_per_row[row_b + 2 : row_b + 3, h0 : h0 + HEAD_CHUNK]
                s3 = score_state_per_row[row_b + 3 : row_b + 4, h0 : h0 + HEAD_CHUNK]
                s4 = score_state_per_row[row_b + 4 : row_b + 5, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK]
                s5 = score_state_per_row[row_b + 5 : row_b + 6, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK]
                s6 = score_state_per_row[row_b + 6 : row_b + 7, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK]
                s7 = score_state_per_row[row_b + 7 : row_b + 8, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK]

                m01 = pl.maximum(s0, s1)
                m23 = pl.maximum(s2, s3)
                m45 = pl.maximum(s4, s5)
                m67 = pl.maximum(s6, s7)
                m0123 = pl.maximum(m01, m23)
                m4567 = pl.maximum(m45, m67)
                s_max = pl.maximum(m0123, m4567)

                d0 = pl.sub(s0, s_max)
                d1 = pl.sub(s1, s_max)
                d2 = pl.sub(s2, s_max)
                d3 = pl.sub(s3, s_max)
                d4 = pl.sub(s4, s_max)
                d5 = pl.sub(s5, s_max)
                d6 = pl.sub(s6, s_max)
                d7 = pl.sub(s7, s_max)
                e0 = pl.exp(d0)
                e1 = pl.exp(d1)
                e2 = pl.exp(d2)
                e3 = pl.exp(d3)
                e4 = pl.exp(d4)
                e5 = pl.exp(d5)
                e6 = pl.exp(d6)
                e7 = pl.exp(d7)

                es01 = pl.add(e0, e1)
                es23 = pl.add(e2, e3)
                es45 = pl.add(e4, e5)
                es67 = pl.add(e6, e7)
                es0123 = pl.add(es01, es23)
                es4567 = pl.add(es45, es67)
                e_sum = pl.add(es0123, es4567)

                kv0 = kv_state_per_row[row_b + 0 : row_b + 1, h0 : h0 + HEAD_CHUNK]
                kv1 = kv_state_per_row[row_b + 1 : row_b + 2, h0 : h0 + HEAD_CHUNK]
                kv2 = kv_state_per_row[row_b + 2 : row_b + 3, h0 : h0 + HEAD_CHUNK]
                kv3 = kv_state_per_row[row_b + 3 : row_b + 4, h0 : h0 + HEAD_CHUNK]
                kv4 = kv_state_per_row[row_b + 4 : row_b + 5, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK]
                kv5 = kv_state_per_row[row_b + 5 : row_b + 6, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK]
                kv6 = kv_state_per_row[row_b + 6 : row_b + 7, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK]
                kv7 = kv_state_per_row[row_b + 7 : row_b + 8, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK]

                w0 = pl.mul(e0, kv0)
                w1 = pl.mul(e1, kv1)
                w2 = pl.mul(e2, kv2)
                w3 = pl.mul(e3, kv3)
                w4 = pl.mul(e4, kv4)
                w5 = pl.mul(e5, kv5)
                w6 = pl.mul(e6, kv6)
                w7 = pl.mul(e7, kv7)
                ws01 = pl.add(w0, w1)
                ws23 = pl.add(w2, w3)
                ws45 = pl.add(w4, w5)
                ws67 = pl.add(w6, w7)
                ws0123 = pl.add(ws01, ws23)
                ws4567 = pl.add(ws45, ws67)
                wsum = pl.add(ws0123, ws4567)

                pooled[b_idx : b_idx + 1, h0 : h0 + HEAD_CHUNK] = pl.div(wsum, e_sum)

            # Stage 3: roll state[:, :ratio] = state[:, ratio:] for THIS batch,
            # inside the same kernel scope so it strictly follows the pool's
            # reads of this batch's slots. A separate state_shift task races
            # ahead of softmax_pool in the scheduler -- treat pool+shift as
            # one atomic unit per batch.
            kv_state_per_row[row_b : row_b + COMPRESS_RATIO, :] = \
                kv_state_per_row[row_b + COMPRESS_RATIO : row_b + 2 * COMPRESS_RATIO, :]
            score_state_per_row[row_b : row_b + COMPRESS_RATIO, :] = \
                score_state_per_row[row_b + COMPRESS_RATIO : row_b + 2 * COMPRESS_RATIO, :]

    kv_state = pl.reshape(kv_state_per_row, [B, STATE_LEN, OUT_DIM])
    score_state = pl.reshape(score_state_per_row, [B, STATE_LEN, OUT_DIM])

    # Stage 4: RMSNorm over HEAD_DIM with BF16 weight (upcast to FP32 for the multiply).
    normed = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
        sq_sum = pl.full([1, B], dtype=pl.FP32, value=0.0)
        for hb in pl.range(HEAD_BLOCKS):
            h0 = hb * HEAD_CHUNK
            pc_sq = pooled[:, h0 : h0 + HEAD_CHUNK]
            sq = pl.mul(pc_sq, pc_sq)
            rs = pl.row_sum(sq)
            rs_2d = pl.reshape(rs, [1, B])
            sq_sum = pl.add(sq_sum, rs_2d)
        var = pl.mul(sq_sum, HEAD_DIM_INV)
        var = pl.add(var, EPS)
        rms = pl.sqrt(var)
        inv = pl.recip(rms)
        inv_rms = pl.reshape(inv, [B, 1])
        for hb in pl.range(HEAD_BLOCKS):
            h0 = hb * HEAD_CHUNK
            pc_norm = pooled[:, h0 : h0 + HEAD_CHUNK]
            nw = pl.cast(norm_w_2d[:, h0 : h0 + HEAD_CHUNK], target_type=pl.FP32)
            scaled = pl.row_expand_mul(pc_norm, inv_rms)
            normed[:, h0 : h0 + HEAD_CHUNK] = pl.col_expand_mul(scaled, nw)

    # Stage 5a: store non-rope head dims as BF16.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="store_nope"):
        nope = normed[:, :NOPE_HEAD_DIM]
        out[:, :NOPE_HEAD_DIM] = pl.cast(nope, target_type=pl.BF16)

    # Stage 5b: half-vector RoPE on the trailing ROPE_HEAD_DIM cols, then store.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_store"):
        x_lo = normed[:, NOPE_HEAD_DIM : NOPE_HEAD_DIM + HALF_RD]
        x_hi = normed[:, NOPE_HEAD_DIM + HALF_RD : NOPE_HEAD_DIM + 2 * HALF_RD]
        cos_fp32 = pl.cast(cos, target_type=pl.FP32)
        sin_fp32 = pl.cast(sin, target_type=pl.FP32)
        lo_cos = pl.col_expand_mul(x_lo, cos_fp32)
        hi_sin = pl.col_expand_mul(x_hi, sin_fp32)
        lo_sin = pl.col_expand_mul(x_lo, sin_fp32)
        hi_cos = pl.col_expand_mul(x_hi, cos_fp32)
        y_lo = pl.sub(lo_cos, hi_sin)
        y_hi = pl.add(lo_sin, hi_cos)
        out[:, NOPE_HEAD_DIM : NOPE_HEAD_DIM + HALF_RD] = pl.cast(y_lo, target_type=pl.BF16)
        out[:, NOPE_HEAD_DIM + HALF_RD : NOPE_HEAD_DIM + 2 * HALF_RD] = pl.cast(y_hi, target_type=pl.BF16)

    return out


@pl.jit
def compressor_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv_state: pl.Out[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
    score_state: pl.Out[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
    out: pl.Out[pl.Tensor[[B, HEAD_DIM], pl.BF16]],
):
    out = compressor(
        x, kv_state, score_state, wkv, wgate, ape, norm_w, cos, sin, hadamard, start_pos, out,
    )
    return out


def golden_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=4 overlap)."""
    import torch

    x = tensors["x"]
    kv_state = tensors["kv_state"]
    score_state = tensors["score_state"]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"].float()
    norm_w = tensors["norm_w"].float()
    cos = tensors["cos"].float()
    sin = tensors["sin"].float()
    hadamard = tensors["hadamard"].float()

    bsz, _, _ = x.shape
    ratio, overlap, rotate, d, rd = COMPRESS_RATIO, OVERLAP, ROTATE, HEAD_DIM, ROPE_HEAD_DIM
    dtype = x.dtype
    x = x.float()
    kv = x.view(bsz, -1) @ wkv.T
    score = x.view(bsz, -1) @ wgate.T

    should_compress = (START_POS + 1) % ratio == 0
    score = score + ape[START_POS % ratio]
    if overlap:
        kv_state[:bsz, ratio + START_POS % ratio] = kv
        score_state[:bsz, ratio + START_POS % ratio] = score
        if should_compress:
            kvs = torch.cat([kv_state[:bsz, :ratio, :d], kv_state[:bsz, ratio:, d:]], dim=1)
            scs = torch.cat([score_state[:bsz, :ratio, :d], score_state[:bsz, ratio:, d:]], dim=1)
            # Decomposed softmax to mirror the kernel's tree reduction (max/sub/exp/sum/div).
            scs_max = scs.amax(dim=1, keepdim=True)
            scs_exp = (scs - scs_max).exp()
            scs_sum = scs_exp.sum(dim=1, keepdim=True)
            kv = (kvs * scs_exp).sum(dim=1, keepdim=True) / scs_sum
            kv_state[:bsz, :ratio] = kv_state[:bsz, ratio:]
            score_state[:bsz, :ratio] = score_state[:bsz, ratio:]

    if not should_compress:
        tensors["out"][:] = torch.zeros(B, HEAD_DIM, dtype=torch.bfloat16)
        return

    kv_c = kv.squeeze(1)
    kv_c = kv_c * torch.rsqrt(kv_c.square().mean(-1, keepdim=True) + EPS) * norm_w

    half_rd = rd // 2
    x_lo = kv_c[..., -rd:-half_rd]
    x_hi = kv_c[..., -half_rd:]
    cos_v, sin_v = cos.view(-1), sin.view(-1)
    y_lo = x_lo * cos_v - x_hi * sin_v
    y_hi = x_lo * sin_v + x_hi * cos_v
    kv_c = torch.cat([kv_c[..., :-rd], y_lo, y_hi], dim=-1)

    if rotate:
        kv_c = (kv_c @ hadamard).to(torch.bfloat16).float()
    else:
        pass

    tensors["out"][:] = kv_c.to(torch.bfloat16)


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import ScalarSpec, TensorSpec

    torch.manual_seed(42)

    def init_x():
        return torch.randn(B, S, D) - 0.5
    def init_kv_state():
        return torch.zeros(B, STATE_LEN, OUT_DIM)
    def init_score_state():
        return torch.full((B, STATE_LEN, OUT_DIM), float("-inf"))
    def init_wkv():
        return (torch.randn(OUT_DIM, D) - 0.5) / (D ** 0.5)
    def init_wgate():
        return (torch.randn(OUT_DIM, D) - 0.5) / (D ** 0.5)
    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.01
    def init_norm_w():
        return torch.ones(HEAD_DIM)
    def init_cos():
        return torch.cos(torch.arange(ROPE_HEAD_DIM // 2).reshape(1, ROPE_HEAD_DIM // 2) * 1e-3)
    def init_sin():
        return torch.sin(torch.arange(ROPE_HEAD_DIM // 2).reshape(1, ROPE_HEAD_DIM // 2) * 1e-3)
    def init_hadamard():
        return torch.eye(HEAD_DIM)
    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_kv_state, is_output=True),
        TensorSpec("score_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_score_state, is_output=True),
        TensorSpec("wkv", [OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cos", [1, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_cos),
        TensorSpec("sin", [1, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_sin),
        TensorSpec("hadamard", [HEAD_DIM, HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        ScalarSpec("start_pos", torch.int32, START_POS),
        TensorSpec("out", [B, HEAD_DIM], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=compressor_test,
        specs=build_tensor_specs(),
        golden_fn=golden_compressor,
        config=RunConfig(
            rtol=1e-3,
            atol=1e-3,
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                runtime_profiling=args.runtime_profiling,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
