# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek-V4 MoE feed-forward (decode).

Corresponds to model.py MoE.forward + Gate.forward + Expert.forward
lines 546-647:

  Gate (score branch, layers 3..n_layers-1, model.py 564-584):
    scores = sqrt(softplus(linear(x, gate_w)))
    indices = (scores + bias).topk(top_k).indices
    weights = scores.gather(1, indices)
    weights = (weights / weights.sum(-1, keepdim=True)) * route_scale

  Gate (hash branch, layers 0..n_hash_layers-1, model.py 576-577):
    indices = tid2eid[input_ids.flatten()]
    weights = scores.gather(1, indices)
    weights = (weights / weights.sum(-1, keepdim=True)) * route_scale

  Expert SwiGLU (model.py 596-606):
    gate = clamp(w1(x).float(), max=swiglu_limit)
    up   = clamp(w3(x).float(), -swiglu_limit, swiglu_limit)
    y    = silu(gate) * up
    if weights is not None: y *= weights
    return w2(y)

  MoE (model.py 630-645):
    weights, indices = gate(x, input_ids)
    y = sum_over_topk(expert[i](x[idx], weights[idx, top])) for i in routed
    y += shared_experts(x)             # no clamp, no weight

The mode is selected by the static `mode` argument to build_*_program.
Real V4 has n_routed_experts=384, moe_inter_dim=3072. The constants below
use a demo-scale (n_routed_experts=8, moe_inter_dim=512) to keep skeleton
tensor allocations tractable; real shapes will be wired in by the runtime.

Skeleton stage: kernel body is TODO; golden is a faithful torch port.
"""


import pypto.language as pl


B            = 16
S            = 1
T            = B * S
D            = 7168

# Real model uses N_EXPERTS=384, MOE_INTER=3072. Demo-scale chosen here so
# build_tensor_specs() does not allocate >50 GB of expert weights.
N_EXPERTS    = 8
MOE_INTER    = 512
TOPK         = 6
N_SHARED     = 1
SWIGLU_LIMIT = 10.0
ROUTE_SCALE  = 2.5
VOCAB        = 129280

# Mode selector for the gate. "score" = topk over learned scores (most layers);
# "hash" = lookup in tid2eid table (first few layers).
MODE         = "score"


def build_deepseek_v4_decode_moe_program():
    @pl.program
    class DeepSeekV4DecodeMoE:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_moe(
            self,
            x:          pl.Tensor[[B, S, D],                         pl.BF16],
            input_ids:  pl.Tensor[[B, S],                            pl.INT64],
            gate_w:     pl.Tensor[[N_EXPERTS, D],                    pl.FP32],
            gate_bias:  pl.Tensor[[N_EXPERTS],                       pl.FP32],
            tid2eid:    pl.Tensor[[VOCAB, TOPK],                     pl.INT32],
            expert_w1:  pl.Tensor[[N_EXPERTS, MOE_INTER, D],         pl.BF16],
            expert_w3:  pl.Tensor[[N_EXPERTS, MOE_INTER, D],         pl.BF16],
            expert_w2:  pl.Tensor[[N_EXPERTS, D, MOE_INTER],         pl.BF16],
            shared_w1:  pl.Tensor[[MOE_INTER, D],                    pl.BF16],
            shared_w3:  pl.Tensor[[MOE_INTER, D],                    pl.BF16],
            shared_w2:  pl.Tensor[[D, MOE_INTER],                    pl.BF16],
            out:        pl.Out[pl.Tensor[[B, S, D],                  pl.BF16]],
        ):
            # TODO: kernel implementation
            return out

    return DeepSeekV4DecodeMoE


def golden_deepseek_v4_decode_moe(tensors):
    """Torch reference, direct port of model.py MoE/Gate/Expert.forward."""
    import torch
    import torch.nn.functional as F

    x          = tensors["x"].float()
    input_ids  = tensors["input_ids"]
    gate_w     = tensors["gate_w"].float()
    gate_bias  = tensors["gate_bias"].float()
    tid2eid    = tensors["tid2eid"]
    w1         = tensors["expert_w1"].float()
    w3         = tensors["expert_w3"].float()
    w2         = tensors["expert_w2"].float()
    sw1        = tensors["shared_w1"].float()
    sw3        = tensors["shared_w3"].float()
    sw2        = tensors["shared_w2"].float()

    x_flat = x.view(-1, D)                                              # [T, D]

    # Gate.forward (model.py 564-584)
    scores = F.softplus(x_flat @ gate_w.T).sqrt()                       # [T, N_EXPERTS]
    original_scores = scores
    if MODE == "score":
        biased = scores + gate_bias
        indices = biased.topk(TOPK, dim=-1).indices                     # [T, TOPK]
    else:  # "hash"
        indices = tid2eid[input_ids.flatten().long()]                   # [T, TOPK]
    weights = original_scores.gather(1, indices)                        # [T, TOPK]
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * ROUTE_SCALE

    # MoE.forward dispatch (model.py 630-645)
    y = torch.zeros_like(x_flat)
    for i in range(N_EXPERTS):
        idx, top = torch.where(indices == i)                            # token id, expert slot id
        if idx.numel() == 0:
            continue
        x_i = x_flat[idx]                                               # [n, D]
        # Expert.forward (model.py 596-606)
        gate = (x_i @ w1[i].T).clamp(max=SWIGLU_LIMIT)
        up   = (x_i @ w3[i].T).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        h = F.silu(gate) * up
        h = h * weights[idx, top].unsqueeze(-1)
        y[idx] += h @ w2[i].T

    # Shared expert: no clamp, no per-token weight
    sh_gate = x_flat @ sw1.T
    sh_up   = x_flat @ sw3.T
    sh = F.silu(sh_gate) * sh_up
    y = y + sh @ sw2.T

    tensors["out"][:] = y.view(B, S, D).to(torch.bfloat16)


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import TensorSpec

    def init_x():
        return torch.randn(B, S, D) * 0.05
    def init_input_ids():
        return torch.randint(0, VOCAB, (B, S), dtype=torch.int64)
    def init_gate_w():
        return torch.randn(N_EXPERTS, D) / D ** 0.5
    def init_gate_bias():
        return torch.zeros(N_EXPERTS)
    def init_tid2eid():
        return torch.randint(0, N_EXPERTS, (VOCAB, TOPK), dtype=torch.int32)
    def init_w1():
        return torch.randn(N_EXPERTS, MOE_INTER, D) / D ** 0.5
    def init_w3():
        return torch.randn(N_EXPERTS, MOE_INTER, D) / D ** 0.5
    def init_w2():
        return torch.randn(N_EXPERTS, D, MOE_INTER) / MOE_INTER ** 0.5
    def init_sw1():
        return torch.randn(MOE_INTER, D) / D ** 0.5
    def init_sw3():
        return torch.randn(MOE_INTER, D) / D ** 0.5
    def init_sw2():
        return torch.randn(D, MOE_INTER) / MOE_INTER ** 0.5

    return [
        TensorSpec("x",         [B, S, D],                  torch.bfloat16, init_value=init_x),
        TensorSpec("input_ids", [B, S],                     torch.int64,    init_value=init_input_ids),
        TensorSpec("gate_w",    [N_EXPERTS, D],             torch.float32,  init_value=init_gate_w),
        TensorSpec("gate_bias", [N_EXPERTS],                torch.float32,  init_value=init_gate_bias),
        TensorSpec("tid2eid",   [VOCAB, TOPK],              torch.int32,    init_value=init_tid2eid),
        TensorSpec("expert_w1", [N_EXPERTS, MOE_INTER, D],  torch.bfloat16, init_value=init_w1),
        TensorSpec("expert_w3", [N_EXPERTS, MOE_INTER, D],  torch.bfloat16, init_value=init_w3),
        TensorSpec("expert_w2", [N_EXPERTS, D, MOE_INTER],  torch.bfloat16, init_value=init_w2),
        TensorSpec("shared_w1", [MOE_INTER, D],             torch.bfloat16, init_value=init_sw1),
        TensorSpec("shared_w3", [MOE_INTER, D],             torch.bfloat16, init_value=init_sw3),
        TensorSpec("shared_w2", [D, MOE_INTER],             torch.bfloat16, init_value=init_sw2),
        TensorSpec("out",       [B, S, D],                  torch.bfloat16, is_output=True),
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
        program=build_deepseek_v4_decode_moe_program(),
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_moe,
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
