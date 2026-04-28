# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 KV Compressor (decode incremental, main attention path: ratio=4, head_dim=512,
rotate=False, overlap=True). Updates state buffers and emits one compressed KV every `ratio` steps."""


import pypto.language as pl


B           = 16                      # demo 4
S           = 1
D           = 4096                    # v4-pro 7168
HEAD_DIM    = 512
ROPE_DIM    = 64
NOPE_DIM    = HEAD_DIM - ROPE_DIM
RATIO       = 4
COFF        = 2                      # 1 + (ratio == 4)
STATE_LEN   = COFF * RATIO           # 8
OUT_DIM     = COFF * HEAD_DIM        # 1024
EPS         = 1e-6


def build_deepseek_v4_decode_compressor_program():
    @pl.program
    class DeepSeekV4DecodeCompressor:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_compressor(
            self,
            x:               pl.Tensor[[B, S, D],                       pl.BF16],
            kv_state:        pl.InOut[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
            score_state:     pl.InOut[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
            wkv:             pl.Tensor[[OUT_DIM, D],                    pl.BF16],
            wgate:           pl.Tensor[[OUT_DIM, D],                    pl.BF16],
            ape:             pl.Tensor[[RATIO, OUT_DIM],                pl.FP32],
            weight:          pl.Tensor[[HEAD_DIM],                      pl.BF16],
            cos:             pl.Tensor[[1, ROPE_DIM],                   pl.BF16],
            sin:             pl.Tensor[[1, ROPE_DIM],                   pl.BF16],
            start_pos:       pl.Tensor[[B],                             pl.INT32],
            should_compress: pl.Tensor[[1],                             pl.INT32],
            out:             pl.Out[pl.Tensor[[B, HEAD_DIM],            pl.BF16]],
        ):
            # TODO: kernel implementation
            return out

    return DeepSeekV4DecodeCompressor


def golden_deepseek_v4_decode_compressor(tensors):
    """Torch reference, direct port of model.py Compressor.forward 343-377 (decode branch)."""
    import torch

    x         = tensors["x"].float()                       # [B, S, D]
    kv_state  = tensors["kv_state"]                        # InOut
    score_state = tensors["score_state"]                   # InOut
    wkv       = tensors["wkv"].float()                     # [OUT_DIM, D]
    wgate     = tensors["wgate"].float()                   # [OUT_DIM, D]
    ape       = tensors["ape"].float()                     # [RATIO, OUT_DIM]
    weight    = tensors["weight"].float()                  # [HEAD_DIM]
    cos       = tensors["cos"].float()                     # [1, ROPE_DIM]
    sin       = tensors["sin"].float()                     # [1, ROPE_DIM]
    start_pos = int(tensors["start_pos"][0].item())        # all batches at same step
    should_compress = bool(tensors["should_compress"][0].item())

    # 1. Project (kv, score) and add positional bias for the current slot
    kv    = (x.view(B, -1) @ wkv.T)                        # [B, OUT_DIM]
    score = (x.view(B, -1) @ wgate.T) + ape[start_pos % RATIO]

    # 2. Update state buffers (overlap branch, model.py 346-348)
    slot = RATIO + start_pos % RATIO
    kv_state[:, slot]    = kv
    score_state[:, slot] = score

    out = torch.zeros(B, HEAD_DIM, dtype=torch.bfloat16)

    if should_compress:
        # 3. Concat overlapping halves: state[:, :ratio, :head_dim] + state[:, ratio:, head_dim:]
        kv_view    = torch.cat([kv_state[:, :RATIO, :HEAD_DIM],
                                kv_state[:, RATIO:, HEAD_DIM:]], dim=1)
        score_view = torch.cat([score_state[:, :RATIO, :HEAD_DIM],
                                score_state[:, RATIO:, HEAD_DIM:]], dim=1)
        kv_c = (kv_view * torch.softmax(score_view, dim=1)).sum(dim=1)   # [B, HEAD_DIM]

        # State shift: state[:ratio] <- state[ratio:]
        kv_state[:, :RATIO]    = kv_state[:, RATIO:]
        score_state[:, :RATIO] = score_state[:, RATIO:]

        # 4. RMSNorm (gamma = weight)
        inv = torch.rsqrt(kv_c.square().mean(-1, keepdim=True) + EPS)
        kv_c = kv_c * inv * weight

        # 5. RoPE on the last rope_dim of head_dim (interleaved pairs)
        x_pair = kv_c[..., NOPE_DIM:].unflatten(-1, (-1, 2))
        x0, x1 = x_pair[..., 0], x_pair[..., 1]
        cos_v = cos.view(-1)
        sin_v = sin.view(-1)
        y0 = x0 * cos_v - x1 * sin_v
        y1 = x0 * sin_v + x1 * cos_v
        kv_c = torch.cat([kv_c[..., :NOPE_DIM],
                          torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

        # 6. FP8 quant simulation on nope dims is identity in skeleton
        out = kv_c.to(torch.bfloat16)

    tensors["out"][:] = out


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import TensorSpec

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
        return torch.randn(RATIO, OUT_DIM) * 0.01
    def init_weight():
        return torch.ones(HEAD_DIM)
    def init_cos():
        return torch.cos(torch.arange(ROPE_DIM).reshape(1, ROPE_DIM) * 1e-3)
    def init_sin():
        return torch.sin(torch.arange(ROPE_DIM).reshape(1, ROPE_DIM) * 1e-3)
    def init_start_pos():
        return torch.tensor([RATIO - 1] * B, dtype=torch.int32)
    def init_should_compress():
        return torch.tensor([1], dtype=torch.int32)

    return [
        TensorSpec("x",               [B, S, D],                  torch.bfloat16, init_value=init_x),
        TensorSpec("kv_state",        [B, STATE_LEN, OUT_DIM],    torch.float32,  init_value=init_kv_state),
        TensorSpec("score_state",     [B, STATE_LEN, OUT_DIM],    torch.float32,  init_value=init_score_state),
        TensorSpec("wkv",             [OUT_DIM, D],               torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate",           [OUT_DIM, D],               torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape",             [RATIO, OUT_DIM],           torch.float32,  init_value=init_ape),
        TensorSpec("weight",          [HEAD_DIM],                 torch.bfloat16, init_value=init_weight),
        TensorSpec("cos",             [1, ROPE_DIM],              torch.bfloat16, init_value=init_cos),
        TensorSpec("sin",             [1, ROPE_DIM],              torch.bfloat16, init_value=init_sin),
        TensorSpec("start_pos",       [B],                        torch.int32,    init_value=init_start_pos),
        TensorSpec("should_compress", [1],                        torch.int32,    init_value=init_should_compress),
        TensorSpec("out",             [B, HEAD_DIM],              torch.bfloat16, is_output=True),
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
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_compressor,
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
