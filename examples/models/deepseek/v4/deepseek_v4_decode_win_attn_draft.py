# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 sliding-window attention (decode, compress_ratio == 0): pure window sparse softmax
with sink and fused inverse RoPE, used by layers without KV compression."""


import pypto.language as pl


B           = 16               # demo 4
S           = 1
T           = B * S
H           = 64               # v4-pro 128
HEAD_DIM    = 512
ROPE_DIM    = 64
NOPE_DIM    = HEAD_DIM - ROPE_DIM
WIN         = 128

BLOCK_SIZE      = 128
ORI_MAX_BLOCKS  = 1                      # WIN==BLOCK_SIZE => 1 block per batch (placeholder)
ORI_BLOCK_NUM   = B * ORI_MAX_BLOCKS

SOFTMAX_SCALE   = HEAD_DIM ** -0.5


def build_deepseek_v4_decode_win_attn_program():
    @pl.program
    class DeepSeekV4DecodeWinAttn:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_win_attn(
            self,
            q:                pl.Tensor[[T, H, HEAD_DIM],                          pl.BF16],
            ori_kv:           pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],  pl.BF16],
            ori_block_table:  pl.Tensor[[B, ORI_MAX_BLOCKS],                       pl.INT32],
            window_topk_idxs: pl.Tensor[[T, WIN],                                  pl.INT32],
            attn_sink:        pl.Tensor[[H],                                       pl.FP32],
            seqused_kv:       pl.Tensor[[B],                                       pl.INT32],
            freqs_cos:        pl.Tensor[[T, ROPE_DIM],                             pl.BF16],
            freqs_sin:        pl.Tensor[[T, ROPE_DIM],                             pl.BF16],
            o:                pl.Out[pl.Tensor[[T, H, HEAD_DIM],                   pl.BF16]],
        ):
            # TODO: kernel implementation
            return o

    return DeepSeekV4DecodeWinAttn


def golden_deepseek_v4_decode_win_attn(tensors):
    """Torch reference: gather window KV via block table, masked-softmax with sink, inverse RoPE."""
    import torch

    q                = tensors["q"].float()
    ori_kv           = tensors["ori_kv"].float()
    ori_block_table  = tensors["ori_block_table"]
    window_topk_idxs = tensors["window_topk_idxs"]
    attn_sink        = tensors["attn_sink"].float()
    cos              = tensors["freqs_cos"].float()
    sin              = tensors["freqs_sin"].float()

    out = torch.zeros(T, H, HEAD_DIM)

    for b in range(B):
        idxs = window_topk_idxs[b]                                         # [WIN]
        valid = idxs >= 0
        valid_idxs = idxs[valid]
        if valid_idxs.numel() == 0:
            continue

        gathered = []
        for raw in valid_idxs.tolist():
            blk_id = int(ori_block_table[b, raw // BLOCK_SIZE].item())
            intra  = raw % BLOCK_SIZE
            gathered.append(ori_kv[blk_id, intra, 0])
        kv_b = torch.stack(gathered, dim=0)                                # [N, HEAD_DIM]

        q_b = q[b]                                                          # [H, HEAD_DIM]
        scores = (q_b @ kv_b.T) * SOFTMAX_SCALE                            # [H, N]
        scores = torch.cat([scores, attn_sink.unsqueeze(-1)], dim=-1)
        probs = torch.softmax(scores, dim=-1)[..., :-1]
        out[b] = probs @ kv_b

    # Inverse RoPE on rope dims
    out_rope = out[..., NOPE_DIM:]
    x_pair = out_rope.unflatten(-1, (-1, 2))
    x0, x1 = x_pair[..., 0], x_pair[..., 1]
    cos_v = cos.unsqueeze(1)
    sin_v = sin.unsqueeze(1)
    y0 = x0 * cos_v + x1 * sin_v
    y1 = -x0 * sin_v + x1 * cos_v
    out = torch.cat([out[..., :NOPE_DIM],
                     torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

    tensors["o"][:] = out.to(torch.bfloat16)


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import TensorSpec

    def init_q():
        return torch.randn(T, H, HEAD_DIM) * 0.05
    def init_ori_kv():
        return torch.randn(ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) * 0.05
    def init_attn_sink():
        return torch.zeros(H)

    def init_ori_block_table():
        tbl = torch.full((B, ORI_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(ORI_MAX_BLOCKS):
                tbl[b, j] = b * ORI_MAX_BLOCKS + j
        return tbl

    def init_window_topk_idxs():
        return torch.arange(WIN, dtype=torch.int32).unsqueeze(0).expand(T, -1).contiguous()

    def init_seqused_kv():
        return torch.tensor([WIN] * B, dtype=torch.int32)
    def init_cos():
        return torch.cos(torch.arange(T * ROPE_DIM).reshape(T, ROPE_DIM) * 1e-3)
    def init_sin():
        return torch.sin(torch.arange(T * ROPE_DIM).reshape(T, ROPE_DIM) * 1e-3)

    return [
        TensorSpec("q",                [T, H, HEAD_DIM],                          torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv",           [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],  torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("ori_block_table",  [B, ORI_MAX_BLOCKS],                       torch.int32,    init_value=init_ori_block_table),
        TensorSpec("window_topk_idxs", [T, WIN],                                  torch.int32,    init_value=init_window_topk_idxs),
        TensorSpec("attn_sink",        [H],                                       torch.float32,  init_value=init_attn_sink),
        TensorSpec("seqused_kv",       [B],                                       torch.int32,    init_value=init_seqused_kv),
        TensorSpec("freqs_cos",        [T, ROPE_DIM],                             torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin",        [T, ROPE_DIM],                             torch.bfloat16, init_value=init_sin),
        TensorSpec("o",                [T, H, HEAD_DIM],                          torch.bfloat16, is_output=True),
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
        program=build_deepseek_v4_decode_win_attn_program(),
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_win_attn,
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
