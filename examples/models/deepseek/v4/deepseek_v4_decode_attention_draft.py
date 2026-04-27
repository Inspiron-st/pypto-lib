# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek-V4 attention sublayer orchestration (decode).

End-to-end wrapper for the attention half of model.py Block.forward
lines 690-694, including the inner Attention.forward (model.py 484-543):

    residual = x_hc                                                       # [B,S,hc,D]
    x_mixed, post, comb = hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base)
    x_norm = rms_norm(x_mixed, attn_norm_w)                               # model.py:692

    # Inner Attention.forward
    q, kv, qr = mla(x_norm, ...)                                          # model.py 496-504
    scatter(ori_kv, ori_block_table, write_slot=start_pos % WIN, kv)      # model.py:530
    if compress_ratio in {4, 128}:
        cmp_out = compressor(x_norm, ..., should_compress, ratio)         # model.py:316-377
        if should_compress: scatter(cmp_kv, cmp_block_table, cache_slot, cmp_out)
        if compress_ratio == 4:
            topk_idxs = indexer(x_norm, qr, ..., should_compress)         # model.py 402-433
        else:
            topk_idxs = static_compress_topk_idxs(...)
        sparse_indices = cat(window_topk_idxs(...), topk_idxs)            # model.py 507-515
        o = cfa(q, ori_kv, ..., cmp_kv, ..., sparse_indices, ...)         # model.py:528 / 533
    else:  # compress_ratio == 0
        o = win_attn(q, ori_kv, ..., window_topk_idxs(...), ...)
    attn_out = o_proj(o, wo_a, wo_b)                                      # model.py 537-542

    # hc_post (model.py 684-687, 694)
    x_after_attn = post.unsqueeze(-1) * attn_out.unsqueeze(-2) \
                 + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)
    return x_after_attn

This program is `FunctionType.Orchestration`: it calls the kernel programs
defined in the sibling files. Skeleton stage: orchestration body is TODO; the
golden runs the full attention sublayer end-to-end in torch by composing
each sibling's golden.
"""


import pypto.language as pl


B               = 16
S               = 1
T               = B * S
# Hidden / Attention
D               = 7168
H               = 128
HEAD_DIM        = 512
ROPE_DIM        = 64
NOPE_DIM        = HEAD_DIM - ROPE_DIM
Q_LORA          = 1536
O_LORA          = 1024
O_GROUPS        = 16
O_GROUP_IN      = H * HEAD_DIM // O_GROUPS
WIN             = 128
EPS             = 1e-6

# Hyper-Connections
HC_MULT          = 4
MIX_HC           = (2 + HC_MULT) * HC_MULT
HC_DIM           = HC_MULT * D
HC_SINKHORN_ITER = 20
HC_EPS           = 1e-6

# Indexer
IDX_HEADS    = 64
IDX_HEAD_DIM = 128
IDX_NOPE     = IDX_HEAD_DIM - ROPE_DIM
IDX_TOPK     = 1024

# Compressor (main path defaults: ratio=4, head_dim=512, rotate=False)
RATIO            = 4
COFF             = 2
STATE_LEN        = COFF * RATIO
CMP_OUT_DIM      = COFF * HEAD_DIM
# Inner compressor for indexer (rotate=True, ratio=4, head_dim=128)
INNER_OUT_DIM    = COFF * IDX_HEAD_DIM

# PagedAttention pools
BLOCK_SIZE       = 128
ORI_MAX_BLOCKS   = 1
ORI_BLOCK_NUM    = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS   = 64
CMP_BLOCK_NUM    = B * CMP_MAX_BLOCKS
IDX_MAX_BLOCKS   = 64
IDX_BLOCK_NUM    = B * IDX_MAX_BLOCKS

SOFTMAX_SCALE    = HEAD_DIM ** -0.5
TOPK             = WIN + IDX_TOPK


def build_deepseek_v4_decode_attention_program():
    @pl.program
    class DeepSeekV4DecodeAttention:
        @pl.function(type=pl.FunctionType.Orchestration)
        def deepseek_v4_decode_attention(
            self,
            x_hc:               pl.Tensor[[B, S, HC_MULT, D],                        pl.BF16],
            # hc_pre weights
            hc_attn_fn:         pl.Tensor[[MIX_HC, HC_DIM],                          pl.FP32],
            hc_attn_scale:      pl.Tensor[[3],                                       pl.FP32],
            hc_attn_base:       pl.Tensor[[MIX_HC],                                  pl.FP32],
            # attn_norm
            attn_norm_w:        pl.Tensor[[D],                                       pl.FP32],
            # mla weights
            wq_a:               pl.Tensor[[D, Q_LORA],                               pl.BF16],
            wq_b:               pl.Tensor[[Q_LORA, H * HEAD_DIM],                    pl.BF16],
            wkv:                pl.Tensor[[D, HEAD_DIM],                             pl.BF16],
            gamma_cq:           pl.Tensor[[Q_LORA],                                  pl.BF16],
            gamma_ckv:          pl.Tensor[[HEAD_DIM],                                pl.BF16],
            rope_cos:           pl.Tensor[[T, ROPE_DIM],                             pl.BF16],
            rope_sin:           pl.Tensor[[T, ROPE_DIM],                             pl.BF16],
            # compressor weights / state (main path: ratio=4, head_dim=512, rotate=False)
            cmp_wkv:            pl.Tensor[[CMP_OUT_DIM, D],                          pl.BF16],
            cmp_wgate:          pl.Tensor[[CMP_OUT_DIM, D],                          pl.BF16],
            cmp_ape:            pl.Tensor[[RATIO, CMP_OUT_DIM],                      pl.FP32],
            cmp_weight:         pl.Tensor[[HEAD_DIM],                                pl.BF16],
            cmp_cos:            pl.Tensor[[1, ROPE_DIM],                             pl.BF16],
            cmp_sin:            pl.Tensor[[1, ROPE_DIM],                             pl.BF16],
            cmp_kv_state:       pl.InOut[pl.Tensor[[B, STATE_LEN, CMP_OUT_DIM],      pl.FP32]],
            cmp_score_state:    pl.InOut[pl.Tensor[[B, STATE_LEN, CMP_OUT_DIM],      pl.FP32]],
            # indexer weights / state
            idx_wq_b:           pl.Tensor[[Q_LORA, IDX_HEADS * IDX_HEAD_DIM],        pl.BF16],
            weights_proj:       pl.Tensor[[D, IDX_HEADS],                            pl.BF16],
            hadamard_q:         pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM],              pl.BF16],
            inner_wkv:          pl.Tensor[[INNER_OUT_DIM, D],                        pl.BF16],
            inner_wgate:        pl.Tensor[[INNER_OUT_DIM, D],                        pl.BF16],
            inner_ape:          pl.Tensor[[RATIO, INNER_OUT_DIM],                    pl.FP32],
            inner_weight:       pl.Tensor[[IDX_HEAD_DIM],                            pl.BF16],
            inner_hadamard:     pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM],              pl.BF16],
            inner_kv_state:     pl.InOut[pl.Tensor[[B, STATE_LEN, INNER_OUT_DIM],    pl.FP32]],
            inner_score_state:  pl.InOut[pl.Tensor[[B, STATE_LEN, INNER_OUT_DIM],    pl.FP32]],
            # PagedAttention pools and tables
            ori_kv:             pl.InOut[pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
            ori_block_table:    pl.Tensor[[B, ORI_MAX_BLOCKS],                       pl.INT32],
            cmp_kv:             pl.InOut[pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
            cmp_block_table:    pl.Tensor[[B, CMP_MAX_BLOCKS],                       pl.INT32],
            idx_kv_cache:       pl.InOut[pl.Tensor[[IDX_BLOCK_NUM, BLOCK_SIZE, IDX_HEAD_DIM], pl.BF16]],
            idx_block_table:    pl.Tensor[[B, IDX_MAX_BLOCKS],                       pl.INT32],
            seqused_kv:         pl.Tensor[[B],                                       pl.INT32],
            # o_proj weights
            wo_a:               pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN],            pl.BF16],
            wo_b:               pl.Tensor[[D, O_GROUPS * O_LORA],                    pl.BF16],
            # cfa / win_attn shared
            attn_sink:          pl.Tensor[[H],                                       pl.FP32],
            # control
            start_pos:          pl.Tensor[[B],                                       pl.INT32],
            should_compress:    pl.Tensor[[1],                                       pl.INT32],
            offset:             pl.Tensor[[1],                                       pl.INT32],
            x_out:              pl.Out[pl.Tensor[[B, S, HC_MULT, D],                 pl.BF16]],
        ):
            # TODO: orchestration body. Pseudo-code:
            #   x_mixed, post, comb = self.hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base)
            #   x_norm = rms_norm(x_mixed, attn_norm_w)
            #   q, kv, qr = self.mla(x_norm, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv)
            #   pl.scatter(ori_kv, ori_block_table, write_slot=start_pos % WIN, kv)
            #   cmp_out = self.compressor(x_norm, cmp_kv_state, cmp_score_state, cmp_wkv, cmp_wgate,
            #                             cmp_ape, cmp_weight, cmp_cos, cmp_sin, start_pos, should_compress)
            #   pl.scatter(cmp_kv, cmp_block_table, cache_slot=start_pos // RATIO, cmp_out)
            #   topk_idxs = self.indexer(x_norm, qr, idx_wq_b, weights_proj, rope_cos, rope_sin,
            #                            hadamard_q, inner_*, inner_kv_state, inner_score_state,
            #                            idx_kv_cache, idx_block_table, seqused_kv,
            #                            start_pos, should_compress, offset)
            #   sparse_indices = pl.cat([window_topk_idxs(start_pos), topk_idxs], dim=-1)
            #   o = self.cfa(q, ori_kv, ori_block_table, cmp_kv, cmp_block_table,
            #                sparse_indices, attn_sink, seqused_kv, rope_cos, rope_sin)
            #   attn_out = self.o_proj(o, wo_a, wo_b)
            #   x_out = post.unsqueeze(-1) * attn_out.unsqueeze(-2) \
            #         + (comb.unsqueeze(-1) * x_hc.unsqueeze(-2)).sum(dim=2)
            return x_out

    return DeepSeekV4DecodeAttention


def golden_deepseek_v4_decode_attention(tensors):
    """End-to-end torch reference for the attention sublayer.

    Composes the goldens of hc_pre, mla, compressor, indexer, cfa, o_proj
    and the inline hc_post. Each sub-step is a faithful port of the
    matching model.py function.
    """
    import torch

    # ---- inputs ----
    x_hc           = tensors["x_hc"].float()                              # [B,S,hc,D]
    hc_attn_fn     = tensors["hc_attn_fn"].float()
    hc_attn_scale  = tensors["hc_attn_scale"].float()
    hc_attn_base   = tensors["hc_attn_base"].float()
    attn_norm_w    = tensors["attn_norm_w"].float()
    wq_a           = tensors["wq_a"].float()
    wq_b           = tensors["wq_b"].float()
    wkv            = tensors["wkv"].float()
    gamma_cq       = tensors["gamma_cq"].float()
    gamma_ckv      = tensors["gamma_ckv"].float()
    rope_cos       = tensors["rope_cos"].float()
    rope_sin       = tensors["rope_sin"].float()
    cmp_wkv        = tensors["cmp_wkv"].float()
    cmp_wgate      = tensors["cmp_wgate"].float()
    cmp_ape        = tensors["cmp_ape"].float()
    cmp_weight     = tensors["cmp_weight"].float()
    cmp_cos        = tensors["cmp_cos"].float()
    cmp_sin        = tensors["cmp_sin"].float()
    cmp_kv_state   = tensors["cmp_kv_state"]
    cmp_score_state = tensors["cmp_score_state"]
    idx_wq_b       = tensors["idx_wq_b"].float()
    weights_proj   = tensors["weights_proj"].float()
    hadamard_q     = tensors["hadamard_q"].float()
    inner_wkv      = tensors["inner_wkv"].float()
    inner_wgate    = tensors["inner_wgate"].float()
    inner_ape      = tensors["inner_ape"].float()
    inner_weight   = tensors["inner_weight"].float()
    inner_hadamard = tensors["inner_hadamard"].float()
    inner_kv_state = tensors["inner_kv_state"]
    inner_score_state = tensors["inner_score_state"]
    ori_kv         = tensors["ori_kv"]
    ori_block_table = tensors["ori_block_table"]
    cmp_kv         = tensors["cmp_kv"]
    cmp_block_table = tensors["cmp_block_table"]
    idx_kv_cache   = tensors["idx_kv_cache"]
    idx_block_table = tensors["idx_block_table"]
    seqused_kv     = tensors["seqused_kv"]
    wo_a           = tensors["wo_a"].float()
    wo_b           = tensors["wo_b"].float()
    attn_sink      = tensors["attn_sink"].float()
    start_pos      = int(tensors["start_pos"][0].item())
    should_compress = bool(tensors["should_compress"][0].item())
    offset         = int(tensors["offset"][0].item())

    # ---- hc_pre (model.py 674-682) ----
    x_flat = x_hc.flatten(2)
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + EPS)
    mixes = (x_flat @ hc_attn_fn.T) * rsqrt
    pre  = torch.sigmoid(mixes[..., :HC_MULT] * hc_attn_scale[0] + hc_attn_base[:HC_MULT]) + HC_EPS
    post = 2 * torch.sigmoid(mixes[..., HC_MULT:HC_MULT * 2] * hc_attn_scale[1] + hc_attn_base[HC_MULT:HC_MULT * 2])
    comb = (mixes[..., HC_MULT * 2:] * hc_attn_scale[2] + hc_attn_base[HC_MULT * 2:]
            ).view(*mixes.shape[:-1], HC_MULT, HC_MULT)
    comb = torch.softmax(comb, dim=-1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    x_mixed = (pre.unsqueeze(-1) * x_hc).sum(dim=2)                       # [B,S,D]

    # ---- attn_norm (model.py 692) ----
    inv = torch.rsqrt(x_mixed.square().mean(-1, keepdim=True) + EPS)
    x_norm = (x_mixed * inv * attn_norm_w).contiguous()                   # [B,S,D]
    x_norm_t = x_norm.view(T, D)

    def apply_rope(x_rope, cos, sin):
        x_pair = x_rope.unflatten(-1, (-1, 2))
        x0, x1 = x_pair[..., 0], x_pair[..., 1]
        cos_ = cos.view(cos.size(0), -1)
        sin_ = sin.view(sin.size(0), -1)
        while cos_.ndim < x0.ndim:
            cos_ = cos_.unsqueeze(-2)
            sin_ = sin_.unsqueeze(-2)
        y0 = x0 * cos_ - x1 * sin_
        y1 = x0 * sin_ + x1 * cos_
        return torch.stack([y0, y1], dim=-1).flatten(-2)

    # ---- mla (model.py 496-504) ----
    qr = (x_norm_t @ wq_a)
    qr_inv = torch.rsqrt(qr.square().mean(-1, keepdim=True) + EPS)
    qr = qr * qr_inv * gamma_cq                                            # [T, Q_LORA]
    q_full = (qr @ wq_b).view(T, H, HEAD_DIM)
    q_full = q_full * torch.rsqrt(q_full.square().mean(-1, keepdim=True) + EPS)
    q_nope = q_full[..., :NOPE_DIM]
    q_rope = apply_rope(q_full[..., NOPE_DIM:], rope_cos, rope_sin)
    q = torch.cat([q_nope, q_rope], dim=-1)                                # [T, H, HEAD_DIM]

    kv_full = x_norm_t @ wkv
    kv_full = kv_full * torch.rsqrt(kv_full.square().mean(-1, keepdim=True) + EPS) * gamma_ckv
    kv_nope = kv_full[..., :NOPE_DIM]
    kv_rope = apply_rope(kv_full[..., NOPE_DIM:].unsqueeze(1), rope_cos, rope_sin).squeeze(1)
    kv = torch.cat([kv_nope, kv_rope], dim=-1)                             # [T, HEAD_DIM]

    # ---- write kv to ori_kv via block_table (model.py:530) ----
    write_slot = start_pos % WIN
    for b in range(B):
        blk_id = int(ori_block_table[b, write_slot // BLOCK_SIZE].item())
        intra  = write_slot % BLOCK_SIZE
        ori_kv[blk_id, intra, 0] = kv[b].to(torch.bfloat16)

    # ---- compressor (main, model.py 343-377 decode branch) ----
    cmp_out = torch.zeros(B, HEAD_DIM, dtype=torch.bfloat16)
    cmp_kv_proj    = (x_norm_t @ cmp_wkv.T)
    cmp_score_proj = (x_norm_t @ cmp_wgate.T) + cmp_ape[start_pos % RATIO]
    slot = RATIO + start_pos % RATIO
    cmp_kv_state[:, slot]    = cmp_kv_proj
    cmp_score_state[:, slot] = cmp_score_proj
    if should_compress:
        kv_view = torch.cat([cmp_kv_state[:, :RATIO, :HEAD_DIM],
                             cmp_kv_state[:, RATIO:, HEAD_DIM:]], dim=1)
        sc_view = torch.cat([cmp_score_state[:, :RATIO, :HEAD_DIM],
                             cmp_score_state[:, RATIO:, HEAD_DIM:]], dim=1)
        kv_c = (kv_view * torch.softmax(sc_view, dim=1)).sum(dim=1)
        cmp_kv_state[:, :RATIO]    = cmp_kv_state[:, RATIO:]
        cmp_score_state[:, :RATIO] = cmp_score_state[:, RATIO:]
        kv_c = kv_c * torch.rsqrt(kv_c.square().mean(-1, keepdim=True) + EPS) * cmp_weight
        kv_c_rope = apply_rope(kv_c[..., NOPE_DIM:].unsqueeze(1), cmp_cos, cmp_sin).squeeze(1)
        kv_c = torch.cat([kv_c[..., :NOPE_DIM], kv_c_rope], dim=-1)
        cmp_out = kv_c.to(torch.bfloat16)

        # Write into cmp_kv pool
        cache_slot = start_pos // RATIO
        for b in range(B):
            blk_id = int(cmp_block_table[b, cache_slot // BLOCK_SIZE].item())
            intra  = cache_slot % BLOCK_SIZE
            cmp_kv[blk_id, intra, 0] = cmp_out[b]

    # ---- indexer (model.py 402-433) ----
    q_idx = (qr @ idx_wq_b).view(T, IDX_HEADS, IDX_HEAD_DIM)
    q_idx_rope = apply_rope(q_idx[..., IDX_NOPE:], rope_cos, rope_sin)
    q_idx = torch.cat([q_idx[..., :IDX_NOPE], q_idx_rope], dim=-1)
    q_idx = (q_idx.view(-1, IDX_HEAD_DIM) @ hadamard_q).view(T, IDX_HEADS, IDX_HEAD_DIM) \
            * (IDX_HEAD_DIM ** -0.5)

    inner_kv = (x_norm_t @ inner_wkv.T)
    inner_score = (x_norm_t @ inner_wgate.T) + inner_ape[start_pos % RATIO]
    inner_kv_state[:, slot]    = inner_kv
    inner_score_state[:, slot] = inner_score
    if should_compress:
        kv_view = torch.cat([inner_kv_state[:, :RATIO, :IDX_HEAD_DIM],
                             inner_kv_state[:, RATIO:, IDX_HEAD_DIM:]], dim=1)
        sc_view = torch.cat([inner_score_state[:, :RATIO, :IDX_HEAD_DIM],
                             inner_score_state[:, RATIO:, IDX_HEAD_DIM:]], dim=1)
        inner_c = (kv_view * torch.softmax(sc_view, dim=1)).sum(dim=1)
        inner_kv_state[:, :RATIO]    = inner_kv_state[:, RATIO:]
        inner_score_state[:, :RATIO] = inner_score_state[:, RATIO:]
        inner_c = inner_c * torch.rsqrt(inner_c.square().mean(-1, keepdim=True) + EPS) * inner_weight
        inner_c = (inner_c @ inner_hadamard) * (IDX_HEAD_DIM ** -0.5)
        cache_slot = start_pos // RATIO
        for b in range(B):
            blk_id = int(idx_block_table[b, cache_slot // BLOCK_SIZE].item())
            intra  = cache_slot % BLOCK_SIZE
            idx_kv_cache[blk_id, intra] = inner_c[b].to(torch.bfloat16)

    end_pos = start_pos + 1
    cache_len = end_pos // RATIO
    if cache_len > 0:
        dense_kv = torch.zeros(B, cache_len, IDX_HEAD_DIM)
        for b in range(B):
            for j in range(cache_len):
                blk_id = int(idx_block_table[b, j // BLOCK_SIZE].item())
                intra  = j % BLOCK_SIZE
                dense_kv[b, j] = idx_kv_cache[blk_id, intra].float()
        weights_idx = (x_norm_t @ weights_proj) * (IDX_HEAD_DIM ** -0.5 * IDX_HEADS ** -0.5)
        weights_idx = weights_idx.view(B, IDX_HEADS)
        score = torch.einsum("thd,btd->bht", q_idx, dense_kv)
        score = (torch.relu(score) * weights_idx.view(B, IDX_HEADS, 1)).sum(dim=1)
        k = min(IDX_TOPK, cache_len)
        _, idx = score.topk(k, dim=-1)
        idx = idx.to(torch.int32) + offset
        pad = torch.full((B, IDX_TOPK - k), -1, dtype=torch.int32)
        cmp_topk_idxs = torch.cat([idx, pad], dim=-1)
    else:
        cmp_topk_idxs = torch.full((T, IDX_TOPK), -1, dtype=torch.int32)

    # ---- window topk indices (model.py:255-265) ----
    win_idxs = torch.arange(WIN, dtype=torch.int32).unsqueeze(0).expand(T, -1).contiguous()
    sparse_indices = torch.cat([win_idxs, cmp_topk_idxs.view(T, IDX_TOPK)], dim=-1)

    # ---- cfa (model.py 528 / 533 + 534) ----
    out_attn = torch.zeros(T, H, HEAD_DIM)
    for b in range(B):
        idxs = sparse_indices[b]
        valid = idxs >= 0
        valid_idxs = idxs[valid]
        if valid_idxs.numel() == 0:
            continue
        gathered = []
        for raw in valid_idxs.tolist():
            if raw < WIN:
                blk_id = int(ori_block_table[b, raw // BLOCK_SIZE].item())
                intra  = raw % BLOCK_SIZE
                gathered.append(ori_kv[blk_id, intra, 0].float())
            else:
                cmp_slot = raw - WIN
                blk_id = int(cmp_block_table[b, cmp_slot // BLOCK_SIZE].item())
                intra  = cmp_slot % BLOCK_SIZE
                gathered.append(cmp_kv[blk_id, intra, 0].float())
        kv_b = torch.stack(gathered, dim=0)
        scores = (q[b] @ kv_b.T) * SOFTMAX_SCALE
        scores = torch.cat([scores, attn_sink.unsqueeze(-1)], dim=-1)
        probs = torch.softmax(scores, dim=-1)[..., :-1]
        out_attn[b] = probs @ kv_b

    # Inverse RoPE on rope dims (model.py:534)
    o_rope = out_attn[..., NOPE_DIM:]
    x_pair = o_rope.unflatten(-1, (-1, 2))
    x0, x1 = x_pair[..., 0], x_pair[..., 1]
    cos_v = rope_cos.unsqueeze(1)
    sin_v = rope_sin.unsqueeze(1)
    y0 = x0 * cos_v + x1 * sin_v
    y1 = -x0 * sin_v + x1 * cos_v
    out_attn = torch.cat([out_attn[..., :NOPE_DIM],
                          torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

    # ---- o_proj (model.py 537-542) ----
    o_g = out_attn.view(T, O_GROUPS, O_GROUP_IN)
    o_r = torch.einsum("tgd,grd->tgr", o_g, wo_a)
    attn_out = (o_r.flatten(1) @ wo_b.T).view(B, S, D)                     # [B,S,D]

    # ---- hc_post (model.py 684-687) ----
    x_after = post.unsqueeze(-1) * attn_out.unsqueeze(-2) \
              + (comb.unsqueeze(-1) * x_hc.unsqueeze(-2)).sum(dim=2)       # [B,S,hc,D]

    tensors["x_out"][:] = x_after.to(torch.bfloat16)


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import TensorSpec

    def init_x_hc():
        return torch.randn(B, S, HC_MULT, D) * 0.05
    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) / HC_DIM ** 0.5
    def init_hc_attn_scale():
        return torch.ones(3) * 0.5
    def init_hc_attn_base():
        return torch.zeros(MIX_HC)
    def init_attn_norm_w():
        return torch.ones(D)
    def init_wq_a():
        return torch.randn(D, Q_LORA) / D ** 0.5
    def init_wq_b():
        return torch.randn(Q_LORA, H * HEAD_DIM) / Q_LORA ** 0.5
    def init_wkv():
        return torch.randn(D, HEAD_DIM) / D ** 0.5
    def init_gamma_cq():
        return torch.ones(Q_LORA)
    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)
    def init_rope_cos():
        return torch.cos(torch.arange(T * ROPE_DIM).reshape(T, ROPE_DIM) * 1e-3)
    def init_rope_sin():
        return torch.sin(torch.arange(T * ROPE_DIM).reshape(T, ROPE_DIM) * 1e-3)
    def init_cmp_wkv():
        return torch.randn(CMP_OUT_DIM, D) / D ** 0.5
    def init_cmp_wgate():
        return torch.randn(CMP_OUT_DIM, D) / D ** 0.5
    def init_cmp_ape():
        return torch.randn(RATIO, CMP_OUT_DIM) * 0.01
    def init_cmp_weight():
        return torch.ones(HEAD_DIM)
    def init_cmp_cos():
        return torch.cos(torch.arange(ROPE_DIM).reshape(1, ROPE_DIM) * 1e-3)
    def init_cmp_sin():
        return torch.sin(torch.arange(ROPE_DIM).reshape(1, ROPE_DIM) * 1e-3)
    def init_cmp_kv_state():
        return torch.zeros(B, STATE_LEN, CMP_OUT_DIM)
    def init_cmp_score_state():
        return torch.full((B, STATE_LEN, CMP_OUT_DIM), float("-inf"))
    def init_idx_wq_b():
        return torch.randn(Q_LORA, IDX_HEADS * IDX_HEAD_DIM) / Q_LORA ** 0.5
    def init_weights_proj():
        return torch.randn(D, IDX_HEADS) / D ** 0.5
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
    def init_ori_kv():
        return torch.randn(ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) * 0.05
    def init_cmp_kv():
        return torch.zeros(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
    def init_idx_kv_cache():
        return torch.zeros(IDX_BLOCK_NUM, BLOCK_SIZE, IDX_HEAD_DIM)

    def init_ori_block_table():
        tbl = torch.full((B, ORI_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(ORI_MAX_BLOCKS):
                tbl[b, j] = b * ORI_MAX_BLOCKS + j
        return tbl

    def init_cmp_block_table():
        tbl = torch.full((B, CMP_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(CMP_MAX_BLOCKS):
                tbl[b, j] = b * CMP_MAX_BLOCKS + j
        return tbl

    def init_idx_block_table():
        tbl = torch.full((B, IDX_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(IDX_MAX_BLOCKS):
                tbl[b, j] = b * IDX_MAX_BLOCKS + j
        return tbl

    def init_seqused_kv():
        return torch.tensor([WIN] * B, dtype=torch.int32)
    def init_wo_a():
        return torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / O_GROUP_IN ** 0.5
    def init_wo_b():
        return torch.randn(D, O_GROUPS * O_LORA) / (O_GROUPS * O_LORA) ** 0.5
    def init_attn_sink():
        return torch.zeros(H)
    def init_start_pos():
        return torch.tensor([RATIO - 1] * B, dtype=torch.int32)
    def init_should_compress():
        return torch.tensor([1], dtype=torch.int32)
    def init_offset():
        return torch.tensor([0], dtype=torch.int32)

    return [
        TensorSpec("x_hc",              [B, S, HC_MULT, D],                              torch.bfloat16, init_value=init_x_hc),
        TensorSpec("hc_attn_fn",        [MIX_HC, HC_DIM],                                torch.float32,  init_value=init_hc_attn_fn),
        TensorSpec("hc_attn_scale",     [3],                                             torch.float32,  init_value=init_hc_attn_scale),
        TensorSpec("hc_attn_base",      [MIX_HC],                                        torch.float32,  init_value=init_hc_attn_base),
        TensorSpec("attn_norm_w",       [D],                                             torch.float32,  init_value=init_attn_norm_w),
        TensorSpec("wq_a",              [D, Q_LORA],                                     torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b",              [Q_LORA, H * HEAD_DIM],                          torch.bfloat16, init_value=init_wq_b),
        TensorSpec("wkv",               [D, HEAD_DIM],                                   torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq",          [Q_LORA],                                        torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv",         [HEAD_DIM],                                      torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("rope_cos",          [T, ROPE_DIM],                                   torch.bfloat16, init_value=init_rope_cos),
        TensorSpec("rope_sin",          [T, ROPE_DIM],                                   torch.bfloat16, init_value=init_rope_sin),
        TensorSpec("cmp_wkv",           [CMP_OUT_DIM, D],                                torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate",         [CMP_OUT_DIM, D],                                torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape",           [RATIO, CMP_OUT_DIM],                            torch.float32,  init_value=init_cmp_ape),
        TensorSpec("cmp_weight",        [HEAD_DIM],                                      torch.bfloat16, init_value=init_cmp_weight),
        TensorSpec("cmp_cos",           [1, ROPE_DIM],                                   torch.bfloat16, init_value=init_cmp_cos),
        TensorSpec("cmp_sin",           [1, ROPE_DIM],                                   torch.bfloat16, init_value=init_cmp_sin),
        TensorSpec("cmp_kv_state",      [B, STATE_LEN, CMP_OUT_DIM],                     torch.float32,  init_value=init_cmp_kv_state),
        TensorSpec("cmp_score_state",   [B, STATE_LEN, CMP_OUT_DIM],                     torch.float32,  init_value=init_cmp_score_state),
        TensorSpec("idx_wq_b",          [Q_LORA, IDX_HEADS * IDX_HEAD_DIM],              torch.bfloat16, init_value=init_idx_wq_b),
        TensorSpec("weights_proj",      [D, IDX_HEADS],                                  torch.bfloat16, init_value=init_weights_proj),
        TensorSpec("hadamard_q",        [IDX_HEAD_DIM, IDX_HEAD_DIM],                    torch.bfloat16, init_value=init_hadamard_q),
        TensorSpec("inner_wkv",         [INNER_OUT_DIM, D],                              torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate",       [INNER_OUT_DIM, D],                              torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape",         [RATIO, INNER_OUT_DIM],                          torch.float32,  init_value=init_inner_ape),
        TensorSpec("inner_weight",      [IDX_HEAD_DIM],                                  torch.bfloat16, init_value=init_inner_weight),
        TensorSpec("inner_hadamard",    [IDX_HEAD_DIM, IDX_HEAD_DIM],                    torch.bfloat16, init_value=init_inner_hadamard),
        TensorSpec("inner_kv_state",    [B, STATE_LEN, INNER_OUT_DIM],                   torch.float32,  init_value=init_inner_kv_state),
        TensorSpec("inner_score_state", [B, STATE_LEN, INNER_OUT_DIM],                   torch.float32,  init_value=init_inner_score_state),
        TensorSpec("ori_kv",            [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],        torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("ori_block_table",   [B, ORI_MAX_BLOCKS],                             torch.int32,    init_value=init_ori_block_table),
        TensorSpec("cmp_kv",            [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],        torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table",   [B, CMP_MAX_BLOCKS],                             torch.int32,    init_value=init_cmp_block_table),
        TensorSpec("idx_kv_cache",      [IDX_BLOCK_NUM, BLOCK_SIZE, IDX_HEAD_DIM],       torch.bfloat16, init_value=init_idx_kv_cache),
        TensorSpec("idx_block_table",   [B, IDX_MAX_BLOCKS],                             torch.int32,    init_value=init_idx_block_table),
        TensorSpec("seqused_kv",        [B],                                             torch.int32,    init_value=init_seqused_kv),
        TensorSpec("wo_a",              [O_GROUPS, O_LORA, O_GROUP_IN],                  torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b",              [D, O_GROUPS * O_LORA],                          torch.bfloat16, init_value=init_wo_b),
        TensorSpec("attn_sink",         [H],                                             torch.float32,  init_value=init_attn_sink),
        TensorSpec("start_pos",         [B],                                             torch.int32,    init_value=init_start_pos),
        TensorSpec("should_compress",   [1],                                             torch.int32,    init_value=init_should_compress),
        TensorSpec("offset",            [1],                                             torch.int32,    init_value=init_offset),
        TensorSpec("x_out",             [B, S, HC_MULT, D],                              torch.bfloat16, is_output=True),
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
        program=build_deepseek_v4_decode_attention_program(),
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_attention,
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
