# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek-V4 Hyper-Connections pre-mix.

Corresponds to model.py Block.hc_pre lines 674-682:
    x = x.flatten(2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base,
                                        hc_mult, sinkhorn_iters, hc_eps)
    y = sum(pre.unsqueeze(-1) * x.view(B, S, hc, D), dim=2)

Same program is used for both attn-pre and ffn-pre, with different weights.
Skeleton stage: kernel body is TODO; golden is a faithful torch port.
"""


import pypto.language as pl


B                = 16
S                = 1
D                = 7168
HC_MULT          = 4
MIX_HC           = (2 + HC_MULT) * HC_MULT
HC_DIM           = HC_MULT * D
HC_SINKHORN_ITER = 20
HC_EPS           = 1e-6
NORM_EPS         = 1e-6


def build_deepseek_v4_decode_hc_pre_program():
    @pl.program
    class DeepSeekV4DecodeHcPre:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_hc_pre(
            self,
            x:        pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
            hc_fn:    pl.Tensor[[MIX_HC, HC_DIM],   pl.FP32],
            hc_scale: pl.Tensor[[3],                pl.FP32],
            hc_base:  pl.Tensor[[MIX_HC],           pl.FP32],
            x_mixed:  pl.Out[pl.Tensor[[B, S, D],            pl.BF16]],
            post:     pl.Out[pl.Tensor[[B, S, HC_MULT],      pl.FP32]],
            comb:     pl.Out[pl.Tensor[[B, S, HC_MULT, HC_MULT], pl.FP32]],
        ):
            # TODO: kernel implementation
            return x_mixed, post, comb

    return DeepSeekV4DecodeHcPre


def golden_deepseek_v4_decode_hc_pre(tensors):
    """Torch reference, direct port of model.py Block.hc_pre 674-682 + hc_split_sinkhorn."""
    import torch

    x        = tensors["x"].float()                        # [B, S, hc, D]
    hc_fn    = tensors["hc_fn"].float()                    # [mix_hc, hc*D]
    hc_scale = tensors["hc_scale"].float()                 # [3]
    hc_base  = tensors["hc_base"].float()                  # [mix_hc]

    shape = x.size()
    x_flat = x.flatten(2)                                  # [B, S, hc*D]
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = (x_flat @ hc_fn.T) * rsqrt                     # [B, S, mix_hc]

    # hc_split_sinkhorn (port of kernel.py 372-427)
    pre = torch.sigmoid(mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post_t = 2 * torch.sigmoid(mixes[..., HC_MULT:HC_MULT * 2] * hc_scale[1]
                               + hc_base[HC_MULT:HC_MULT * 2])
    comb_t = (mixes[..., HC_MULT * 2:] * hc_scale[2] + hc_base[HC_MULT * 2:]
              ).view(*mixes.shape[:-1], HC_MULT, HC_MULT)

    # First step: row-softmax then col-normalize, with eps after softmax
    comb_t = torch.softmax(comb_t, dim=-1) + HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    # Sinkhorn iterations
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = comb_t / (comb_t.sum(-1, keepdim=True) + HC_EPS)
        comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)

    y = (pre.unsqueeze(-1) * x.view(shape)).sum(dim=2)     # [B, S, D]

    tensors["x_mixed"][:] = y.to(torch.bfloat16)
    tensors["post"][:]    = post_t
    tensors["comb"][:]    = comb_t


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x():
        return torch.randn(B, S, HC_MULT, D) * 0.1
    def init_hc_fn():
        return torch.randn(MIX_HC, HC_DIM) / (HC_DIM ** 0.5)
    def init_hc_scale():
        return torch.ones(3) * 0.5
    def init_hc_base():
        return torch.zeros(MIX_HC)

    return [
        TensorSpec("x",        [B, S, HC_MULT, D],       torch.bfloat16, init_value=init_x),
        TensorSpec("hc_fn",    [MIX_HC, HC_DIM],         torch.float32,  init_value=init_hc_fn),
        TensorSpec("hc_scale", [3],                      torch.float32,  init_value=init_hc_scale),
        TensorSpec("hc_base",  [MIX_HC],                 torch.float32,  init_value=init_hc_base),
        TensorSpec("x_mixed",  [B, S, D],                torch.bfloat16, is_output=True),
        TensorSpec("post",     [B, S, HC_MULT],          torch.float32,  is_output=True),
        TensorSpec("comb",     [B, S, HC_MULT, HC_MULT], torch.float32,  is_output=True),
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
        program=build_deepseek_v4_decode_hc_pre_program(),
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_hc_pre,
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
