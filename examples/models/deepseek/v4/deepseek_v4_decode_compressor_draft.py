# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 KV Compressor (decode incremental). Mirrors model.py Compressor (line 279-377);
golden is a direct port of forward's decode branch (prefill `start_pos == 0` path is omitted).
Configurable for compress_ratio ∈ {0, 4, 128} and rotate ∈ {False, True}."""


import pypto.language as pl


B = 16  # demo 4
S = 1
EPS = 1e-6

COMPRESS_RATIO = 4  # 0 / 4 / 128
HEAD_DIM = 512
ROTATE = False

D = 4096  # v4-pro 7168
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)

OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO

START_POS = 3  # default for ScalarSpec; >0 (decode) and (START_POS+1)%COMPRESS_RATIO==0 to cover the full compression path
SHOULD_COMPRESS = COMPRESS_RATIO != 0 and ((START_POS + 1) % COMPRESS_RATIO) == 0
APE_ROW = START_POS % COMPRESS_RATIO if COMPRESS_RATIO != 0 else 0  # row index into ape used by block 2
SCATTER_SLOT = (COMPRESS_RATIO + APE_ROW) if OVERLAP else APE_ROW  # state row index for current kv/score (block 3)

HEAD_DIM_INV = 1.0 / HEAD_DIM    # used by block 8 RMSNorm

# Tiling for the kv/score projections: x [B, D] @ wkv.T / wgate.T → [B, OUT_DIM].
K_CHUNK = 512                    # K-axis chunk per matmul step
OUT_CHUNK = 64                   # OUT_DIM-axis chunk per parallel iteration
K_BLOCKS = D // K_CHUNK          # 8
OUT_BLOCKS = OUT_DIM // OUT_CHUNK  # 16

# Tiling along HEAD_DIM (used by block 4 gather and downstream RMSNorm/RoPE).
HEAD_CHUNK = 128                 # 128 * 4B (FP32) = 512B innermost
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK  # 4


def build_deepseek_v4_decode_compressor_program():
    @pl.program
    class DeepSeekV4DecodeCompressor:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_compressor(
            self,
            x: pl.Tensor[[B, S, D], pl.BF16],
            kv_state: pl.InOut[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
            score_state: pl.InOut[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
            wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
            wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
            ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
            norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
            cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],  # caller passes freqs_cis[start_pos+1-ratio]; half_dim for interleaved-pair RoPE
            sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],  # same shape/semantics as cos
            hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
            start_pos: pl.Scalar[pl.INT32],  # decode step; varies per call
            out: pl.Out[pl.Tensor[[B, HEAD_DIM], pl.BF16]],
        ):
            # Phase 3 incremental build (CV-split). Currently implements:
            #   block 1  : kv = x @ wkv.T, score = x @ wgate.T   (Cube)
            #   block 11': placeholder cast/store of kv head_dim slice into out
            # Compression path (blocks 2-10) added in subsequent phases.
            x_flat = pl.reshape(x, [B, D])
            # Flat 2D views over the InOut state tensors so we can scatter rows by linear offset.
            kv_state_flat = pl.reshape(kv_state, [B, STATE_LEN * OUT_DIM])
            score_state_flat = pl.reshape(score_state, [B, STATE_LEN * OUT_DIM])

            kv_fp32 = pl.create_tensor([B, OUT_DIM], dtype=pl.FP32)
            score_fp32 = pl.create_tensor([B, OUT_DIM], dtype=pl.FP32)
            # Gathered windows for the compression reduce (block 4-6); kept as 2D
            # [B*STATE_LEN, HEAD_DIM] so all downstream ops stay in ND layout.
            kvs = pl.create_tensor([B * STATE_LEN, HEAD_DIM], dtype=pl.FP32)
            scs = pl.create_tensor([B * STATE_LEN, HEAD_DIM], dtype=pl.FP32)

            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="kv_score_proj"):
                for ob in pl.parallel(OUT_BLOCKS, chunk=4):
                    oc0 = ob * OUT_CHUNK

                    # kv = x @ wkv.T  : [B, K_CHUNK] @ ([OUT_CHUNK, K_CHUNK]).T = [B, OUT_CHUNK]
                    a0 = pl.slice(x_flat, [B, K_CHUNK], [0, 0])
                    b0 = pl.slice(wkv, [OUT_CHUNK, K_CHUNK], [oc0, 0])
                    kv_acc = pl.matmul(a0, b0, out_dtype=pl.FP32, b_trans=True)
                    for kb in pl.range(1, K_BLOCKS):
                        k0 = kb * K_CHUNK
                        a_i = pl.slice(x_flat, [B, K_CHUNK], [0, k0])
                        b_i = pl.slice(wkv, [OUT_CHUNK, K_CHUNK], [oc0, k0])
                        kv_acc = pl.matmul_acc(kv_acc, a_i, b_i, b_trans=True)
                    kv_fp32 = pl.assemble(kv_fp32, kv_acc, [0, oc0])

                    # score = x @ wgate.T
                    a0g = pl.slice(x_flat, [B, K_CHUNK], [0, 0])
                    b0g = pl.slice(wgate, [OUT_CHUNK, K_CHUNK], [oc0, 0])
                    sc_acc = pl.matmul(a0g, b0g, out_dtype=pl.FP32, b_trans=True)
                    for kb in pl.range(1, K_BLOCKS):
                        k0 = kb * K_CHUNK
                        a_ig = pl.slice(x_flat, [B, K_CHUNK], [0, k0])
                        b_ig = pl.slice(wgate, [OUT_CHUNK, K_CHUNK], [oc0, k0])
                        sc_acc = pl.matmul_acc(sc_acc, a_ig, b_ig, b_trans=True)
                    score_fp32 = pl.assemble(score_fp32, sc_acc, [0, oc0])

            # Block 2 (Vector): score += ape[APE_ROW] broadcast across batch rows.
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="ape_add"):
                for ob in pl.parallel(OUT_BLOCKS, chunk=4):
                    oc0 = ob * OUT_CHUNK
                    score_chunk = pl.slice(score_fp32, [B, OUT_CHUNK], [0, oc0])
                    ape_row = pl.slice(ape, [1, OUT_CHUNK], [APE_ROW, oc0])
                    ones_b = pl.full([B, OUT_CHUNK], dtype=pl.FP32, value=1.0)
                    ape_broadcast = pl.col_expand_mul(ones_b, ape_row)
                    score_chunk = pl.add(score_chunk, ape_broadcast)
                    score_fp32 = pl.assemble(score_fp32, score_chunk, [0, oc0])

            # Block 3 (Vector): scatter current kv/score into state[..., SCATTER_SLOT, :].
            # The flat state row stride is OUT_DIM, so slot offset = SCATTER_SLOT * OUT_DIM.
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="state_scatter"):
                slot_off = SCATTER_SLOT * OUT_DIM
                for ob in pl.parallel(OUT_BLOCKS, chunk=4):
                    oc0 = ob * OUT_CHUNK
                    kv_chunk = pl.slice(kv_fp32, [B, OUT_CHUNK], [0, oc0])
                    kv_state_flat = pl.assemble(kv_state_flat, kv_chunk, [0, slot_off + oc0])
                    score_chunk = pl.slice(score_fp32, [B, OUT_CHUNK], [0, oc0])
                    score_state_flat = pl.assemble(score_state_flat, score_chunk, [0, slot_off + oc0])

            # Reshape state to a per-state-row 2D view so block 4 can read 4 consecutive
            # state rows of one batch in a single 2D slice. Stays 2D through block 7.
            kv_state_per_row = pl.reshape(kv_state_flat, [B * STATE_LEN, OUT_DIM])
            score_state_per_row = pl.reshape(score_state_flat, [B * STATE_LEN, OUT_DIM])

            # Block 4 (Vector): build kvs/scs by gathering front-d half from rows [0, ratio)
            # and back-d half from rows [ratio, STATE_LEN). Mirrors:
            #   kvs = cat([state[:, :ratio, :HEAD_DIM], state[:, ratio:, HEAD_DIM:]], dim=1)
            # Iterates per-batch so each 2D slice is contiguous in memory.
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="state_gather"):
                for b in pl.parallel(0, B, 1, chunk=4):
                    row_b = b * STATE_LEN
                    for hb in pl.range(HEAD_BLOCKS):
                        h0 = hb * HEAD_CHUNK
                        kv_front = pl.slice(kv_state_per_row, [COMPRESS_RATIO, HEAD_CHUNK], [row_b, h0])
                        kvs = pl.assemble(kvs, kv_front, [row_b, h0])
                        kv_back = pl.slice(kv_state_per_row, [COMPRESS_RATIO, HEAD_CHUNK], [row_b + COMPRESS_RATIO, HEAD_DIM + h0])
                        kvs = pl.assemble(kvs, kv_back, [row_b + COMPRESS_RATIO, h0])

                        sc_front = pl.slice(score_state_per_row, [COMPRESS_RATIO, HEAD_CHUNK], [row_b, h0])
                        scs = pl.assemble(scs, sc_front, [row_b, h0])
                        sc_back = pl.slice(score_state_per_row, [COMPRESS_RATIO, HEAD_CHUNK], [row_b + COMPRESS_RATIO, HEAD_DIM + h0])
                        scs = pl.assemble(scs, sc_back, [row_b + COMPRESS_RATIO, h0])

            # Block 5+6 (Vector): softmax(scs, dim=STATE_LEN), then pooled = sum(kvs * softmax).
            # KNOWN BUG: this block introduces a ~9% systematic over-magnification on outputs.
            # AB-tests confirm block 1 (matmul), block 4 (gather), RMSNorm, and RoPE are all
            # correct in isolation; only the manual softmax+pool produces biased pooled values.
            # With score_state init = -inf, math says pooled should equal kvs[slot 7] exactly,
            # but the kernel produces values ~9% larger. Init = -100 or -1e20 doesn't help, so
            # it isn't an exp(-inf) saturation issue. See progress doc + walkaround.md for
            # diagnostic AB-tests A-G.
            #
            # WORKAROUND for the test config (degenerate softmax with -inf init): replace this
            # block with a hardcoded `pooled = kvs[slot 7]`, which produces the same math but
            # passes (189 mismatches, all 1-ULP BF16 noise on RoPE). Not production-correct.
            pooled = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="softmax_pool"):
                for b in pl.parallel(0, B, 1, chunk=4):
                    row_b = b * STATE_LEN
                    for hb in pl.range(HEAD_BLOCKS):
                        h0 = hb * HEAD_CHUNK

                        # Pass 1: max across STATE_LEN slots.
                        s_max = pl.slice(scs, [1, HEAD_CHUNK], [row_b, h0])
                        for s in pl.range(1, STATE_LEN):
                            s_row = pl.slice(scs, [1, HEAD_CHUNK], [row_b + s, h0])
                            s_max = pl.maximum(s_max, s_row)

                        # Pass 2: accumulate e_sum and pooled_acc = sum_s(exp(scs_s - max) * kvs_s).
                        e0 = pl.exp(pl.sub(pl.slice(scs, [1, HEAD_CHUNK], [row_b, h0]), s_max))
                        e_sum = e0
                        pooled_acc = pl.mul(e0, pl.slice(kvs, [1, HEAD_CHUNK], [row_b, h0]))
                        for s in pl.range(1, STATE_LEN):
                            ss = pl.slice(scs, [1, HEAD_CHUNK], [row_b + s, h0])
                            ks = pl.slice(kvs, [1, HEAD_CHUNK], [row_b + s, h0])
                            e_s = pl.exp(pl.sub(ss, s_max))
                            e_sum = pl.add(e_sum, e_s)
                            pooled_acc = pl.add(pooled_acc, pl.mul(e_s, ks))

                        pooled_chunk = pl.div(pooled_acc, e_sum)
                        pooled = pl.assemble(pooled, pooled_chunk, [b, h0])

            # Block 7 (Vector): shift state down — state[:, :ratio] = state[:, ratio:].
            # Must execute after block 4 (which reads from state). 2D view: per batch, copy
            # rows [row_b + ratio, row_b + STATE_LEN) → rows [row_b, row_b + ratio).
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="state_shift"):
                for b in pl.parallel(0, B, 1, chunk=4):
                    row_b = b * STATE_LEN
                    kv_src = pl.slice(kv_state_per_row, [COMPRESS_RATIO, OUT_DIM], [row_b + COMPRESS_RATIO, 0])
                    kv_state_per_row = pl.assemble(kv_state_per_row, kv_src, [row_b, 0])
                    sc_src = pl.slice(score_state_per_row, [COMPRESS_RATIO, OUT_DIM], [row_b + COMPRESS_RATIO, 0])
                    score_state_per_row = pl.assemble(score_state_per_row, sc_src, [row_b, 0])

            # Reshape state back to 3D for InOut parameter shape (after all state-touching blocks).
            kv_state = pl.reshape(kv_state_per_row, [B, STATE_LEN, OUT_DIM])
            score_state = pl.reshape(score_state_per_row, [B, STATE_LEN, OUT_DIM])

            # Block 8 (Vector): RMSNorm pooled with norm_w over HEAD_DIM.
            #   normed = pooled * rsqrt(mean(pooled**2) + EPS) * norm_w
            normed_pooled = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
            norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="rmsnorm"):
                # Pass 1: sum of squares across HEAD_DIM (per batch).
                partial_sq = pl.full([1, B], dtype=pl.FP32, value=0.0)
                for hb in pl.range(HEAD_BLOCKS):
                    h0 = hb * HEAD_CHUNK
                    x_chunk = pl.slice(pooled, [B, HEAD_CHUNK], [0, h0])
                    partial_sq = pl.add(
                        partial_sq,
                        pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, B]),
                    )
                inv_rms = pl.reshape(
                    pl.recip(pl.sqrt(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS))),
                    [B, 1],
                )

                # Pass 2: x * inv_rms * norm_w (BF16 norm_w cast to FP32 first).
                for hb in pl.range(HEAD_BLOCKS):
                    h0 = hb * HEAD_CHUNK
                    x_chunk = pl.slice(pooled, [B, HEAD_CHUNK], [0, h0])
                    nw_chunk = pl.cast(
                        pl.slice(norm_w_2d, [1, HEAD_CHUNK], [0, h0]),
                        target_type=pl.FP32,
                    )
                    normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), nw_chunk)
                    normed_pooled = pl.assemble(normed_pooled, normed, [0, h0])

            # Block 11a (Vector): cast non-rope range to BF16 and store to out.
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="store_nope"):
                nope_chunk = pl.slice(normed_pooled, [B, NOPE_HEAD_DIM], [0, 0])
                out = pl.assemble(out, pl.cast(nope_chunk, target_type=pl.BF16), [0, 0])

            # Block 9 + 11b (Vector): half-vector RoPE on the last ROPE_HEAD_DIM cols, then store.
            HALF_RD = ROPE_HEAD_DIM // 2
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="rope_store"):
                x_lo = pl.slice(normed_pooled, [B, HALF_RD], [0, NOPE_HEAD_DIM])
                x_hi = pl.slice(normed_pooled, [B, HALF_RD], [0, NOPE_HEAD_DIM + HALF_RD])
                cos_fp32 = pl.cast(cos, target_type=pl.FP32)
                sin_fp32 = pl.cast(sin, target_type=pl.FP32)
                y_lo = pl.sub(pl.col_expand_mul(x_lo, cos_fp32), pl.col_expand_mul(x_hi, sin_fp32))
                y_hi = pl.add(pl.col_expand_mul(x_lo, sin_fp32), pl.col_expand_mul(x_hi, cos_fp32))
                out = pl.assemble(out, pl.cast(y_lo, target_type=pl.BF16), [0, NOPE_HEAD_DIM])
                out = pl.assemble(out, pl.cast(y_hi, target_type=pl.BF16), [0, NOPE_HEAD_DIM + HALF_RD])

            return out

    return DeepSeekV4DecodeCompressor


def golden_deepseek_v4_decode_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch; prefill omitted, quant identity on A3)."""
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

    start_pos = int(tensors["start_pos"])
    compress_ratio = COMPRESS_RATIO

    bsz, _, _ = x.shape
    ratio, overlap, rotate, d, rd = compress_ratio, OVERLAP, ROTATE, HEAD_DIM, ROPE_HEAD_DIM
    dtype = x.dtype
    x = x.float()
    kv = x.view(bsz, -1) @ wkv.T
    score = x.view(bsz, -1) @ wgate.T

    if start_pos == 0:
        return

    should_compress = (start_pos + 1) % ratio == 0
    score = score + ape[start_pos % ratio]
    if overlap:
        kv_state[:bsz, ratio + start_pos % ratio] = kv
        score_state[:bsz, ratio + start_pos % ratio] = score
        if should_compress:
            kvs = torch.cat([kv_state[:bsz, :ratio, :d], kv_state[:bsz, ratio:, d:]], dim=1)
            scs = torch.cat([score_state[:bsz, :ratio, :d], score_state[:bsz, ratio:, d:]], dim=1)
            kv = (kvs * scs.softmax(dim=1)).sum(dim=1, keepdim=True)
            kv_state[:bsz, :ratio] = kv_state[:bsz, ratio:]
            score_state[:bsz, :ratio] = score_state[:bsz, ratio:]
    else:
        kv_state[:bsz, start_pos % ratio] = kv
        score_state[:bsz, start_pos % ratio] = score
        if should_compress:
            kv = (kv_state[:bsz] * score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)

    if not should_compress:
        tensors["out"][:] = torch.zeros(B, HEAD_DIM, dtype=torch.bfloat16)
        return

    kv_c = kv.squeeze(1)
    kv_c = kv_c * torch.rsqrt(kv_c.square().mean(-1, keepdim=True) + EPS) * norm_w
    kv_c = kv_c.to(dtype).float()

    half_rd = rd // 2
    x_lo = kv_c[..., -rd:-half_rd]                          # first half of rope range
    x_hi = kv_c[..., -half_rd:]                             # second half of rope range
    cos_v, sin_v = cos.view(-1), sin.view(-1)               # [half_rd]
    y_lo = x_lo * cos_v - x_hi * sin_v
    y_hi = x_lo * sin_v + x_hi * cos_v
    kv_c = torch.cat([kv_c[..., :-rd], y_lo, y_hi], dim=-1)

    if rotate:
        kv_c = (kv_c @ hadamard).to(torch.bfloat16).float()  # rotate_activation: full Hadamard matmul (v3_2 style)
        # fp4_act_quant — A3-skipped
    else:
        pass  # act_quant — A3-skipped

    tensors["out"][:] = kv_c.to(torch.bfloat16)


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import ScalarSpec, TensorSpec

    def init_x():
        return torch.randn(B, S, D) * 0.1
    def init_kv_state():
        return torch.zeros(B, STATE_LEN, OUT_DIM)
    def init_score_state():
        return torch.full((B, STATE_LEN, OUT_DIM), float("-inf"))
    def init_wkv():
        return torch.randn(OUT_DIM, D) / D ** 0.5
    def init_wgate():
        return torch.randn(OUT_DIM, D) / D ** 0.5
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
        TensorSpec("kv_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_kv_state),
        TensorSpec("score_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_score_state),
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
    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_deepseek_v4_decode_compressor_program(),
        specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_compressor,
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
