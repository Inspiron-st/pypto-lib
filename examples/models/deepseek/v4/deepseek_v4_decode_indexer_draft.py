# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek-V4 lightning indexer (decode).

Corresponds to model.py Indexer.forward lines 402-433. The full Indexer pipeline
is fused into a single program here:

    q = wq_b(qr).unflatten(-1, (idx_n_heads, idx_head_dim))
    apply_rotary_emb(q[..., -rope_dim:], freqs_cis)
    q = rotate_activation(q)             # Hadamard
    fp4_act_quant(q, ...)                # FP4 quant simulation
    self.compressor(x, start_pos)        # writes idx_kv_cache (rotate=True path)
    weights = weights_proj(x) * (softmax_scale * idx_n_heads ** -0.5)
    score = einsum("bshd,btd->bsht", q, idx_kv_cache[:bsz, :end_pos // ratio])
    score = (relu(score) * weights.unsqueeze(-1)).sum(dim=2)
    topk_idxs = score.topk(min(idx_topk, end_pos // ratio), dim=-1)[1]
    topk_idxs += offset

The internal Compressor uses head_dim=128, rotate=True (model.py:398). For
skeleton clarity, the inner-compressor state buffers are exposed as InOut
inputs so this program is self-contained per call.

Skeleton stage: kernel body is TODO; golden is a faithful torch port.
"""


import pypto.language as pl


B            = 16
S            = 1
T            = B * S
D            = 7168
Q_LORA       = 1536
ROPE_DIM     = 64
EPS          = 1e-6

# Indexer params
IDX_HEADS    = 64
IDX_HEAD_DIM = 128
IDX_NOPE     = IDX_HEAD_DIM - ROPE_DIM
IDX_TOPK     = 1024

# Inner compressor (rotate=True, ratio=4, head_dim=IDX_HEAD_DIM)
RATIO        = 4
COFF         = 2                          # 1 + (RATIO == 4)
STATE_LEN    = COFF * RATIO               # 8
INNER_OUT_DIM = COFF * IDX_HEAD_DIM       # 256

# Paged KV cache for compressed indexer KV
BLOCK_SIZE      = 128
IDX_MAX_BLOCKS  = 64                      # per-batch cap (placeholder)
IDX_BLOCK_NUM   = B * IDX_MAX_BLOCKS      # global pool size (placeholder)


def build_deepseek_v4_decode_indexer_program():
    @pl.program
    class DeepSeekV4DecodeIndexer:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_indexer(
            self,
            x:               pl.Tensor[[B, S, D],                        pl.BF16],
            qr:              pl.Tensor[[T, Q_LORA],                      pl.BF16],
            wq_b:            pl.Tensor[[Q_LORA, IDX_HEADS * IDX_HEAD_DIM], pl.BF16],
            weights_proj:    pl.Tensor[[D, IDX_HEADS],                   pl.BF16],
            cos:             pl.Tensor[[T, ROPE_DIM],                    pl.BF16],
            sin:             pl.Tensor[[T, ROPE_DIM],                    pl.BF16],
            hadamard_q:      pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM],     pl.BF16],
            # Inner compressor weights / state
            inner_wkv:       pl.Tensor[[INNER_OUT_DIM, D],               pl.BF16],
            inner_wgate:     pl.Tensor[[INNER_OUT_DIM, D],               pl.BF16],
            inner_ape:       pl.Tensor[[RATIO, INNER_OUT_DIM],           pl.FP32],
            inner_weight:    pl.Tensor[[IDX_HEAD_DIM],                   pl.BF16],
            inner_hadamard:  pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM],     pl.BF16],
            inner_kv_state:  pl.InOut[pl.Tensor[[B, STATE_LEN, INNER_OUT_DIM], pl.FP32]],
            inner_score_state: pl.InOut[pl.Tensor[[B, STATE_LEN, INNER_OUT_DIM], pl.FP32]],
            # PA-backed compressed indexer KV cache
            idx_kv_cache:    pl.InOut[pl.Tensor[[IDX_BLOCK_NUM, BLOCK_SIZE, IDX_HEAD_DIM], pl.BF16]],
            idx_block_table: pl.Tensor[[B, IDX_MAX_BLOCKS],              pl.INT32],
            seqused_kv:      pl.Tensor[[B],                              pl.INT32],
            start_pos:       pl.Tensor[[B],                              pl.INT32],
            should_compress: pl.Tensor[[1],                              pl.INT32],
            offset:          pl.Tensor[[1],                              pl.INT32],
            topk_idxs:       pl.Out[pl.Tensor[[T, IDX_TOPK],             pl.INT32]],
        ):
            # TODO: kernel implementation
            return topk_idxs

    return DeepSeekV4DecodeIndexer


def golden_deepseek_v4_decode_indexer(tensors):
    """Torch reference, direct port of model.py Indexer.forward 402-433."""
    import torch

    x            = tensors["x"].float()                                   # [B, S, D]
    qr           = tensors["qr"].float()                                  # [T, Q_LORA]
    wq_b         = tensors["wq_b"].float()                                # [Q_LORA, IDX_HEADS*IDX_HEAD_DIM]
    weights_proj = tensors["weights_proj"].float()                        # [D, IDX_HEADS]
    cos          = tensors["cos"].float()                                 # [T, ROPE_DIM]
    sin          = tensors["sin"].float()                                 # [T, ROPE_DIM]
    hadamard_q   = tensors["hadamard_q"].float()                          # [IDX_HEAD_DIM, IDX_HEAD_DIM]

    inner_wkv          = tensors["inner_wkv"].float()
    inner_wgate        = tensors["inner_wgate"].float()
    inner_ape          = tensors["inner_ape"].float()
    inner_weight       = tensors["inner_weight"].float()
    inner_hadamard     = tensors["inner_hadamard"].float()
    inner_kv_state     = tensors["inner_kv_state"]
    inner_score_state  = tensors["inner_score_state"]

    idx_kv_cache    = tensors["idx_kv_cache"]
    idx_block_table = tensors["idx_block_table"]
    seqused_kv      = tensors["seqused_kv"]
    start_pos       = int(tensors["start_pos"][0].item())
    should_compress = bool(tensors["should_compress"][0].item())
    offset          = int(tensors["offset"][0].item())

    # 1. q = wq_b(qr).view(T, IDX_HEADS, IDX_HEAD_DIM)
    q = (qr @ wq_b).view(T, IDX_HEADS, IDX_HEAD_DIM)

    # 2. RoPE on rope dims
    x_pair = q[..., IDX_NOPE:].unflatten(-1, (-1, 2))
    x0, x1 = x_pair[..., 0], x_pair[..., 1]
    cos_v = cos.unsqueeze(1)                                              # [T, 1, ROPE_DIM/2]? – broadcast
    sin_v = sin.unsqueeze(1)
    y0 = x0 * cos_v - x1 * sin_v
    y1 = x0 * sin_v + x1 * cos_v
    q = torch.cat([q[..., :IDX_NOPE], torch.stack([y0, y1], -1).flatten(-2)], dim=-1)

    # 3. Hadamard rotate q (FP4 quant simulation: identity)
    q = (q.view(-1, IDX_HEAD_DIM) @ hadamard_q).view(T, IDX_HEADS, IDX_HEAD_DIM) \
        * (IDX_HEAD_DIM ** -0.5)

    # 4. Run inner Compressor (rotate=True, ratio=4, head_dim=IDX_HEAD_DIM)
    #    State updates and (optional) compression write back to idx_kv_cache.
    kv    = (x.view(B, -1) @ inner_wkv.T)                                 # [B, INNER_OUT_DIM]
    score = (x.view(B, -1) @ inner_wgate.T) + inner_ape[start_pos % RATIO]
    slot  = RATIO + start_pos % RATIO
    inner_kv_state[:, slot]    = kv
    inner_score_state[:, slot] = score

    if should_compress:
        kv_view    = torch.cat([inner_kv_state[:, :RATIO, :IDX_HEAD_DIM],
                                inner_kv_state[:, RATIO:, IDX_HEAD_DIM:]], dim=1)
        score_view = torch.cat([inner_score_state[:, :RATIO, :IDX_HEAD_DIM],
                                inner_score_state[:, RATIO:, IDX_HEAD_DIM:]], dim=1)
        kv_c = (kv_view * torch.softmax(score_view, dim=1)).sum(dim=1)    # [B, IDX_HEAD_DIM]
        # State shift
        inner_kv_state[:, :RATIO]    = inner_kv_state[:, RATIO:]
        inner_score_state[:, :RATIO] = inner_score_state[:, RATIO:]
        # RMSNorm
        inv = torch.rsqrt(kv_c.square().mean(-1, keepdim=True) + EPS)
        kv_c = kv_c * inv * inner_weight
        # Hadamard rotate (inner) — FP4 quant simulation: identity
        kv_c = (kv_c @ inner_hadamard) * (IDX_HEAD_DIM ** -0.5)

        # Write into PA cache at the target slot for each batch
        cache_slot = start_pos // RATIO
        for b in range(B):
            blk_id = int(idx_block_table[b, cache_slot // BLOCK_SIZE].item())
            intra  = cache_slot % BLOCK_SIZE
            idx_kv_cache[blk_id, intra] = kv_c[b].to(torch.bfloat16)

    # 5. Build the per-batch dense view of idx_kv_cache up to end_pos // RATIO
    end_pos_per_b = (start_pos + 1)            # decode: same for all batches in skeleton
    cache_len = end_pos_per_b // RATIO

    if cache_len <= 0:
        topk_idxs = torch.full((T, IDX_TOPK), -1, dtype=torch.int32)
        tensors["topk_idxs"][:] = topk_idxs
        return

    dense_kv = torch.zeros(B, cache_len, IDX_HEAD_DIM)
    for b in range(B):
        for j in range(cache_len):
            blk_id = int(idx_block_table[b, j // BLOCK_SIZE].item())
            intra  = j % BLOCK_SIZE
            dense_kv[b, j] = idx_kv_cache[blk_id, intra].float()

    # 6. weights = weights_proj(x) * softmax_scale * idx_n_heads ** -0.5
    softmax_scale = IDX_HEAD_DIM ** -0.5
    weights = (x.view(B, -1) @ weights_proj) * (softmax_scale * IDX_HEADS ** -0.5)
    weights = weights.view(T, IDX_HEADS)

    # 7. score = einsum("bshd,btd->bsht", q, dense_kv)  (here s=1)
    score = torch.einsum("thd,btd->bht", q, dense_kv)                     # [B, IDX_HEADS, cache_len]
    score = (torch.relu(score) * weights.view(B, IDX_HEADS, 1)).sum(dim=1)  # [B, cache_len]

    # 8. topk over cache_len, padded with -1 to IDX_TOPK
    k = min(IDX_TOPK, cache_len)
    _, idx = score.topk(k, dim=-1)                                        # [B, k]
    idx = idx.to(torch.int32) + offset
    pad = torch.full((B, IDX_TOPK - k), -1, dtype=torch.int32)
    topk_idxs = torch.cat([idx, pad], dim=-1)                             # [B=T, IDX_TOPK]

    tensors["topk_idxs"][:] = topk_idxs


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import TensorSpec

    def init_x():
        return torch.randn(B, S, D) * 0.1
    def init_qr():
        return torch.randn(T, Q_LORA) * 0.1
    def init_wq_b():
        return torch.randn(Q_LORA, IDX_HEADS * IDX_HEAD_DIM) / Q_LORA ** 0.5
    def init_weights_proj():
        return torch.randn(D, IDX_HEADS) / D ** 0.5
    def init_cos():
        return torch.cos(torch.arange(T * ROPE_DIM).reshape(T, ROPE_DIM) * 1e-3)
    def init_sin():
        return torch.sin(torch.arange(T * ROPE_DIM).reshape(T, ROPE_DIM) * 1e-3)
    def init_hadamard_q():
        return torch.eye(IDX_HEAD_DIM)
    def init_inner_wkv():
        return torch.randn(INNER_OUT_DIM, D) / D ** 0.5
    def init_inner_wgate():
        return torch.randn(INNER_OUT_DIM, D) / D ** 0.5
    def init_inner_ape():
        return torch.randn(RATIO, INNER_OUT_DIM) * 0.01
    def init_inner_weight():
        return torch.ones(IDX_HEAD_DIM)
    def init_inner_hadamard():
        return torch.eye(IDX_HEAD_DIM)
    def init_inner_kv_state():
        return torch.zeros(B, STATE_LEN, INNER_OUT_DIM)
    def init_inner_score_state():
        return torch.full((B, STATE_LEN, INNER_OUT_DIM), float("-inf"))
    def init_idx_kv_cache():
        return torch.zeros(IDX_BLOCK_NUM, BLOCK_SIZE, IDX_HEAD_DIM)

    def init_idx_block_table():
        # Each batch gets a contiguous slice of blocks; placeholder layout.
        tbl = torch.full((B, IDX_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(IDX_MAX_BLOCKS):
                tbl[b, j] = b * IDX_MAX_BLOCKS + j
        return tbl

    def init_seqused_kv():
        return torch.tensor([RATIO * 2] * B, dtype=torch.int32)
    def init_start_pos():
        return torch.tensor([RATIO - 1] * B, dtype=torch.int32)
    def init_should_compress():
        return torch.tensor([1], dtype=torch.int32)
    def init_offset():
        return torch.tensor([0], dtype=torch.int32)

    return [
        TensorSpec("x",                 [B, S, D],                                    torch.bfloat16, init_value=init_x),
        TensorSpec("qr",                [T, Q_LORA],                                  torch.bfloat16, init_value=init_qr),
        TensorSpec("wq_b",              [Q_LORA, IDX_HEADS * IDX_HEAD_DIM],           torch.bfloat16, init_value=init_wq_b),
        TensorSpec("weights_proj",      [D, IDX_HEADS],                               torch.bfloat16, init_value=init_weights_proj),
        TensorSpec("cos",               [T, ROPE_DIM],                                torch.bfloat16, init_value=init_cos),
        TensorSpec("sin",               [T, ROPE_DIM],                                torch.bfloat16, init_value=init_sin),
        TensorSpec("hadamard_q",        [IDX_HEAD_DIM, IDX_HEAD_DIM],                 torch.bfloat16, init_value=init_hadamard_q),
        TensorSpec("inner_wkv",         [INNER_OUT_DIM, D],                           torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate",       [INNER_OUT_DIM, D],                           torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape",         [RATIO, INNER_OUT_DIM],                       torch.float32,  init_value=init_inner_ape),
        TensorSpec("inner_weight",      [IDX_HEAD_DIM],                               torch.bfloat16, init_value=init_inner_weight),
        TensorSpec("inner_hadamard",    [IDX_HEAD_DIM, IDX_HEAD_DIM],                 torch.bfloat16, init_value=init_inner_hadamard),
        TensorSpec("inner_kv_state",    [B, STATE_LEN, INNER_OUT_DIM],                torch.float32,  init_value=init_inner_kv_state),
        TensorSpec("inner_score_state", [B, STATE_LEN, INNER_OUT_DIM],                torch.float32,  init_value=init_inner_score_state),
        TensorSpec("idx_kv_cache",      [IDX_BLOCK_NUM, BLOCK_SIZE, IDX_HEAD_DIM],    torch.bfloat16, init_value=init_idx_kv_cache),
        TensorSpec("idx_block_table",   [B, IDX_MAX_BLOCKS],                          torch.int32,    init_value=init_idx_block_table),
        TensorSpec("seqused_kv",        [B],                                          torch.int32,    init_value=init_seqused_kv),
        TensorSpec("start_pos",         [B],                                          torch.int32,    init_value=init_start_pos),
        TensorSpec("should_compress",   [1],                                          torch.int32,    init_value=init_should_compress),
        TensorSpec("offset",            [1],                                          torch.int32,    init_value=init_offset),
        TensorSpec("topk_idxs",         [T, IDX_TOPK],                                torch.int32,    is_output=True),
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
        program=build_deepseek_v4_decode_indexer_program(),
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_indexer,
        config=RunConfig(
            rtol=3e-3,
            atol=3e-3,
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
