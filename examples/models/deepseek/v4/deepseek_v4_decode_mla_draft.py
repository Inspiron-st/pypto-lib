# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek-V4 single-token decode MLA prolog.

Corresponds to model.py Attention.forward lines 496-504:
    qr = q_norm(wq_a(x))
    q  = wq_b(qr).unflatten(-1, (n_local_heads, head_dim))
    q *= rsqrt(q.square().mean(-1, keepdim=True) + eps)        # per-head RMSNorm
    apply_rotary_emb(q[..., -rope_dim:], freqs_cis)
    kv = wkv(x)
    kv = kv_norm(kv)
    apply_rotary_emb(kv[..., -rope_dim:], freqs_cis)

Skeleton stage: kernel body is TODO; golden is a faithful torch port.
"""


import pypto.language as pl


# Decode batch / seq
B           = 16
S           = 1
T           = B * S
# Hidden / Attention
D           = 7168
H           = 128
HEAD_DIM    = 512
ROPE_DIM    = 64
NOPE_DIM    = HEAD_DIM - ROPE_DIM
Q_LORA      = 1536
EPS         = 1e-6


def build_deepseek_v4_decode_mla_program():
    @pl.program
    class DeepSeekV4DecodeMla:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_mla(
            self,
            token_x:   pl.Tensor[[T, D],                pl.BF16],
            wq_a:      pl.Tensor[[D, Q_LORA],           pl.BF16],
            wq_b:      pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.BF16],
            wkv:       pl.Tensor[[D, HEAD_DIM],         pl.BF16],
            rope_cos:  pl.Tensor[[T, ROPE_DIM],         pl.BF16],
            rope_sin:  pl.Tensor[[T, ROPE_DIM],         pl.BF16],
            gamma_cq:  pl.Tensor[[Q_LORA],              pl.BF16],
            gamma_ckv: pl.Tensor[[HEAD_DIM],            pl.BF16],
            q:         pl.Out[pl.Tensor[[T, H, HEAD_DIM], pl.BF16]],
            kv:        pl.Out[pl.Tensor[[T, HEAD_DIM],    pl.BF16]],
            qr:        pl.Out[pl.Tensor[[T, Q_LORA],      pl.BF16]],
        ):
            # TODO: kernel implementation
            return q, kv, qr

    return DeepSeekV4DecodeMla


def golden_deepseek_v4_decode_mla(tensors):
    """Torch reference, direct port of model.py Attention.forward 496-504."""
    import torch

    token_x   = tensors["token_x"].float()
    wq_a      = tensors["wq_a"].float()
    wq_b      = tensors["wq_b"].float()
    wkv       = tensors["wkv"].float()
    rope_cos  = tensors["rope_cos"].float()
    rope_sin  = tensors["rope_sin"].float()
    gamma_cq  = tensors["gamma_cq"].float()
    gamma_ckv = tensors["gamma_ckv"].float()

    def rms_norm(x, gamma, eps=EPS):
        inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        return x * inv * gamma

    def apply_rope(x_rope, cos, sin):
        # x_rope: [T, ..., ROPE_DIM] interleaved pairs
        x_pair = x_rope.unflatten(-1, (-1, 2))
        x0, x1 = x_pair[..., 0], x_pair[..., 1]
        cos_ = cos.view(*([1] * (x_rope.ndim - 2)), cos.size(0), -1)
        sin_ = sin.view(*([1] * (x_rope.ndim - 2)), sin.size(0), -1)
        # Broadcast across heads if any
        while cos_.ndim < x0.ndim:
            cos_ = cos_.unsqueeze(-2)
            sin_ = sin_.unsqueeze(-2)
        y0 = x0 * cos_ - x1 * sin_
        y1 = x0 * sin_ + x1 * cos_
        return torch.stack([y0, y1], dim=-1).flatten(-2)

    # Q path
    qr_out = rms_norm(token_x @ wq_a, gamma_cq)                     # [T, Q_LORA]
    q_full = (qr_out @ wq_b).view(T, H, HEAD_DIM)                   # [T, H, HEAD_DIM]
    inv = torch.rsqrt(q_full.square().mean(-1, keepdim=True) + EPS)
    q_full = q_full * inv                                           # per-head RMSNorm (no gamma)
    q_nope = q_full[..., :NOPE_DIM]
    q_rope = apply_rope(q_full[..., NOPE_DIM:], rope_cos, rope_sin)
    q_out = torch.cat([q_nope, q_rope], dim=-1)

    # KV path
    kv_full = rms_norm(token_x @ wkv, gamma_ckv)                    # [T, HEAD_DIM]
    kv_nope = kv_full[..., :NOPE_DIM]
    kv_rope_in = kv_full[..., NOPE_DIM:].unsqueeze(1)               # add a pseudo head dim
    kv_rope = apply_rope(kv_rope_in, rope_cos, rope_sin).squeeze(1)
    kv_out = torch.cat([kv_nope, kv_rope], dim=-1)

    tensors["q"][:]  = q_out.to(torch.bfloat16)
    tensors["kv"][:] = kv_out.to(torch.bfloat16)
    tensors["qr"][:] = qr_out.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_token_x():
        return torch.randn(T, D) * 0.02
    def init_wq_a():
        return torch.randn(D, Q_LORA) / (D ** 0.5)
    def init_wq_b():
        return torch.randn(Q_LORA, H * HEAD_DIM) / (Q_LORA ** 0.5)
    def init_wkv():
        return torch.randn(D, HEAD_DIM) / (D ** 0.5)
    def init_cos():
        return torch.cos(torch.arange(T * ROPE_DIM).reshape(T, ROPE_DIM) * 1e-3)
    def init_sin():
        return torch.sin(torch.arange(T * ROPE_DIM).reshape(T, ROPE_DIM) * 1e-3)
    def init_gamma_cq():
        return torch.ones(Q_LORA)
    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)

    return [
        TensorSpec("token_x",   [T, D],                torch.bfloat16, init_value=init_token_x),
        TensorSpec("wq_a",      [D, Q_LORA],           torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b",      [Q_LORA, H * HEAD_DIM], torch.bfloat16, init_value=init_wq_b),
        TensorSpec("wkv",       [D, HEAD_DIM],         torch.bfloat16, init_value=init_wkv),
        TensorSpec("rope_cos",  [T, ROPE_DIM],         torch.bfloat16, init_value=init_cos),
        TensorSpec("rope_sin",  [T, ROPE_DIM],         torch.bfloat16, init_value=init_sin),
        TensorSpec("gamma_cq",  [Q_LORA],              torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM],            torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("q",         [T, H, HEAD_DIM],      torch.bfloat16, is_output=True),
        TensorSpec("kv",        [T, HEAD_DIM],         torch.bfloat16, is_output=True),
        TensorSpec("qr",        [T, Q_LORA],           torch.bfloat16, is_output=True),
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
        program=build_deepseek_v4_decode_mla_program(),
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_mla,
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
