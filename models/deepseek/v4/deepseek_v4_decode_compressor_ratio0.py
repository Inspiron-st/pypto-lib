# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 KV Compressor (decode incremental, ratio=0).

No compression. Always outputs zeros. The kernel and state tensors still
exist to match the Compressor interface, but no computation is performed."""


import pypto.language as pl


B = 16
S = 1
EPS = 1e-6

COMPRESS_RATIO = 0
HEAD_DIM = 512
ROTATE = False

D = 4096
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
OVERLAP = False
COFF = 1

OUT_DIM = COFF * HEAD_DIM   # 512
STATE_LEN = 1

START_POS = 0

HEAD_CHUNK = 128
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK  # 4


@pl.jit.inline
def deepseek_v4_decode_compressor(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv_state: pl.Tensor[[B, 1, HEAD_DIM], pl.FP32],
    score_state: pl.Tensor[[B, 1, HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    ape: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
    out: pl.Tensor[[B, HEAD_DIM], pl.BF16],
):
    with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk], name_hint="zeros"):
        for ob in pl.parallel(HEAD_BLOCKS, chunk=4):
            h0 = ob * HEAD_CHUNK
            zero_fp32 = pl.full([B, HEAD_CHUNK], dtype=pl.FP32, value=0.0)
            out = pl.assemble(out, pl.cast(zero_fp32, target_type=pl.BF16), [0, h0])
    return out


@pl.jit
def deepseek_v4_decode_compressor_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv_state: pl.Out[pl.Tensor[[B, 1, HEAD_DIM], pl.FP32]],
    score_state: pl.Out[pl.Tensor[[B, 1, HEAD_DIM], pl.FP32]],
    wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    ape: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
    out: pl.Out[pl.Tensor[[B, HEAD_DIM], pl.BF16]],
):
    out = deepseek_v4_decode_compressor(
        x, kv_state, score_state, wkv, wgate, ape, norm_w, cos, sin, hadamard, start_pos, out,
    )
    return out


def golden_deepseek_v4_decode_compressor(tensors):
    """Torch reference for ratio=0 variant (always zeros)."""
    import torch
    tensors["out"][:] = torch.zeros(B, HEAD_DIM, dtype=torch.bfloat16)


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import ScalarSpec, TensorSpec

    def init_x():
        return torch.randn(B, S, D) - 0.5
    def init_kv_state():
        return torch.zeros(B, 1, HEAD_DIM)
    def init_score_state():
        return torch.full((B, 1, HEAD_DIM), float("-inf"))
    def init_wkv():
        return (torch.randn(HEAD_DIM, D) - 0.5) / (D ** 0.5)
    def init_wgate():
        return (torch.randn(HEAD_DIM, D) - 0.5) / (D ** 0.5)
    def init_ape():
        return torch.randn(1, HEAD_DIM) * 0.01
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
        TensorSpec("kv_state", [B, 1, HEAD_DIM], torch.float32, init_value=init_kv_state, is_output=True),
        TensorSpec("score_state", [B, 1, HEAD_DIM], torch.float32, init_value=init_score_state, is_output=True),
        TensorSpec("wkv", [HEAD_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [HEAD_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [1, HEAD_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cos", [1, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_cos),
        TensorSpec("sin", [1, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_sin),
        TensorSpec("hadamard", [HEAD_DIM, HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        ScalarSpec("start_pos", torch.int32, 0),
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
        fn=deepseek_v4_decode_compressor_test,
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
