# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B decode Scope 1 — input RMSNorm + Q/K/V projection.

Standalone test for the RMSNorm + projection scope of the Qwen3-32B decode layer,
with parameters aligned to qwen3_32b_decode_tilelet.py.

For each batch element:
  1. Compute RMSNorm of input hidden states.
  2. Project to Q (hidden_size), K (kv_hidden), V (kv_hidden).

Hardware TILELET / TILE sizing (at default HIDDEN=5120, KV_HIDDEN=1024):
  * Partial sum [BATCH_TILE, 1]         FP32 = [4,1]*4 = 16 B
  * Norm factor [BATCH_TILE, 1]         FP32 = [4,1]*4 = 16 B
  * Q/K/V accumulator [BATCH_TILE, OUT] FP32 = [4,64]*4 = 1 KB (max 2 KB)
  * Weight tiles [K_CHUNK, OUT]         BF16 = [128,64]*2 = 16 KB = MAX
"""
from __future__ import annotations

import pypto.language as pl

BATCH = 16
MAX_SEQ = 4096
HIDDEN = 5120
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
INTERMEDIATE = 25600

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

# Vector TILELET budget (2 KB = 2048 B, FP32 = 4 B/elem):
K_CHUNK = 128
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
MLP_OUT_CHUNK = 64
BATCH_TILE = 16


def build_decode_projection_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    hidden = hidden_size
    kv_hidden = num_kv_heads * head_dim
    hidden_blocks = hidden // K_CHUNK
    q_out_blocks = hidden // Q_OUT_CHUNK
    kv_out_blocks = kv_hidden // KV_OUT_CHUNK

    @pl.program
    class DecodeProjectionProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def decode_projection(
            self,
            hidden_states: pl.Tensor[[batch, hidden], pl.BF16],
            input_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            wq: pl.Tensor[[hidden, hidden], pl.BF16],
            wk: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            wv: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            q_proj: pl.Out[pl.Tensor[[batch, hidden], pl.BF16]],
            k_proj: pl.Out[pl.Tensor[[batch, kv_hidden], pl.BF16]],
            v_proj: pl.Out[pl.Tensor[[batch, kv_hidden], pl.BF16]],
        ) -> tuple[
            pl.Tensor[[batch, hidden], pl.BF16],
            pl.Tensor[[batch, kv_hidden], pl.BF16],
            pl.Tensor[[batch, kv_hidden], pl.BF16],
        ]:
            for b0 in pl.range(0, batch, BATCH_TILE):
                normed_tile = pl.create_tensor([BATCH_TILE, hidden], dtype=pl.FP32)

                # Stage 1: RMSNorm + apply weights (vector ops only).
                with pl.incore():
                    partial_sq_flat = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    partial_sq = pl.reshape(partial_sq_flat, [BATCH_TILE, 1])
                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        partial_sq = pl.add(partial_sq, pl.row_sum(pl.mul(x_chunk, x_chunk)))
                    inv_rms_tile = pl.rsqrt(pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS))

                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms_tile), gamma)
                        normed_tile = pl.assemble(normed_tile, normed, [0, k0])

                # Stage 2: Q projection.
                for ob in pl.range(q_out_blocks):
                    q0 = ob * Q_OUT_CHUNK

                    with pl.incore():
                        # vec: init accumulator.
                        q_acc = pl.full([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32, value=0.0)

                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        normed_bf16_chunk = pl.create_tensor([BATCH_TILE, K_CHUNK], dtype=pl.BF16)
                        with pl.incore():
                            # vec: cast normed to BF16.
                            normed_chunk = pl.slice(normed_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            normed_bf16_chunk = pl.assemble(
                                normed_bf16_chunk, pl.cast(normed_chunk, target_type=pl.BF16), [0, 0],
                            )

                        q_matmul_out = pl.create_tensor([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                        with pl.incore():
                            # cube: matmul.
                            wq_chunk = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [k0, q0])
                            q_matmul_out = pl.matmul(normed_bf16_chunk, wq_chunk, out_dtype=pl.FP32)

                        with pl.incore():
                            # vec: accumulate.
                            q_acc = pl.add(q_acc, q_matmul_out)

                    with pl.incore():
                        # vec: cast result.
                        q_acc_bf16 = pl.cast(q_acc, target_type=pl.BF16)
                    q_proj = pl.assemble(q_proj, q_acc_bf16, [b0, q0])

                # Stage 3: K/V projection.
                for ob in pl.range(kv_out_blocks):
                    kv0 = ob * KV_OUT_CHUNK

                    with pl.incore():
                        # vec: init accumulators.
                        k_acc = pl.full([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        v_acc = pl.full([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32, value=0.0)

                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        normed_bf16_chunk = pl.create_tensor([BATCH_TILE, K_CHUNK], dtype=pl.BF16)
                        with pl.incore():
                            # vec: cast normed to BF16.
                            normed_chunk = pl.slice(normed_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            normed_bf16_chunk = pl.assemble(
                                normed_bf16_chunk, pl.cast(normed_chunk, target_type=pl.BF16), [0, 0],
                            )

                        k_matmul_out = pl.create_tensor([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                        with pl.incore():
                            # cube: K matmul.
                            wk_chunk = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                            k_matmul_out = pl.matmul(normed_bf16_chunk, wk_chunk, out_dtype=pl.FP32)

                        v_matmul_out = pl.create_tensor([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                        with pl.incore():
                            # cube: V matmul.
                            wv_chunk = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                            v_matmul_out = pl.matmul(normed_bf16_chunk, wv_chunk, out_dtype=pl.FP32)

                        with pl.incore():
                            # vec: accumulate both.
                            k_acc = pl.add(k_acc, k_matmul_out)
                            v_acc = pl.add(v_acc, v_matmul_out)

                    with pl.incore():
                        # vec: cast results.
                        k_acc_bf16 = pl.cast(k_acc, target_type=pl.BF16)
                        v_acc_bf16 = pl.cast(v_acc, target_type=pl.BF16)
                    k_proj = pl.assemble(k_proj, k_acc_bf16, [b0, kv0])
                    v_proj = pl.assemble(v_proj, v_acc_bf16, [b0, kv0])

            return q_proj, k_proj, v_proj

    return DecodeProjectionProgram


def build_tensor_specs(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    import torch
    from pypto.runtime import TensorSpec

    kv_hidden = num_kv_heads * head_dim

    return [
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("input_rms_weight", [1, hidden_size], torch.float32, init_value=torch.randn),
        TensorSpec("wq", [hidden_size, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wk", [hidden_size, kv_hidden], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wv", [hidden_size, kv_hidden], torch.bfloat16, init_value=torch.randn),
        TensorSpec("q_proj", [batch, hidden_size], torch.bfloat16, is_output=True),
        TensorSpec("k_proj", [batch, kv_hidden], torch.bfloat16, is_output=True),
        TensorSpec("v_proj", [batch, kv_hidden], torch.bfloat16, is_output=True),
    ]


def golden_decode_projection(tensors, params):
    """PyTorch reference matching kernel BF16 precision path.

    Tile-by-tile computation to match the kernel's FP32 accumulation across
    K_CHUNK blocks, with BF16 cast on normed chunks before each matmul.
    """
    import torch

    hidden_states = tensors["hidden_states"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]

    batch = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    kv_hidden = wk.shape[1]

    q_proj = torch.zeros(batch, hidden_size, dtype=torch.bfloat16)
    k_proj = torch.zeros(batch, kv_hidden, dtype=torch.bfloat16)
    v_proj = torch.zeros(batch, kv_hidden, dtype=torch.bfloat16)

    for b0 in range(0, batch, BATCH_TILE):
        b_end = min(b0 + BATCH_TILE, batch)
        x_tile = hidden_states[b0:b_end, :].float()

        # RMSNorm: chunked squared sum.
        sq_sum = torch.zeros(b_end - b0, 1, dtype=torch.float32)
        for k0 in range(0, hidden_size, K_CHUNK):
            x_chunk = x_tile[:, k0:k0 + K_CHUNK]
            sq_sum = sq_sum + (x_chunk ** 2).sum(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(sq_sum / hidden_size + EPS)
        normed = x_tile * inv_rms * input_rms_weight.float()

        # Q projection: chunked matmul with FP32 accumulation.
        for q0 in range(0, hidden_size, Q_OUT_CHUNK):
            q_acc = torch.zeros(b_end - b0, Q_OUT_CHUNK, dtype=torch.float32)
            for k0 in range(0, hidden_size, K_CHUNK):
                normed_chunk = normed[:, k0:k0 + K_CHUNK].bfloat16()
                wq_chunk = wq[k0:k0 + K_CHUNK, q0:q0 + Q_OUT_CHUNK]
                q_acc = q_acc + torch.matmul(normed_chunk, wq_chunk).float()
            q_proj[b0:b_end, q0:q0 + Q_OUT_CHUNK] = q_acc.bfloat16()

        # K/V projection: chunked matmul with FP32 accumulation.
        for kv0 in range(0, kv_hidden, KV_OUT_CHUNK):
            k_acc = torch.zeros(b_end - b0, KV_OUT_CHUNK, dtype=torch.float32)
            v_acc = torch.zeros(b_end - b0, KV_OUT_CHUNK, dtype=torch.float32)
            for k0 in range(0, hidden_size, K_CHUNK):
                normed_chunk = normed[:, k0:k0 + K_CHUNK].bfloat16()
                wk_chunk = wk[k0:k0 + K_CHUNK, kv0:kv0 + KV_OUT_CHUNK]
                wv_chunk = wv[k0:k0 + K_CHUNK, kv0:kv0 + KV_OUT_CHUNK]
                k_acc = k_acc + torch.matmul(normed_chunk, wk_chunk).float()
                v_acc = v_acc + torch.matmul(normed_chunk, wv_chunk).float()
            k_proj[b0:b_end, kv0:kv0 + KV_OUT_CHUNK] = k_acc.bfloat16()
            v_proj[b0:b_end, kv0:kv0 + KV_OUT_CHUNK] = v_acc.bfloat16()

    tensors["q_proj"][:] = q_proj
    tensors["k_proj"][:] = k_proj
    tensors["v_proj"][:] = v_proj


def compile_and_run(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    platform: str = "a5",
    device_id: int = 0,
    dump_passes: bool = True,
    enable_profiling: bool = False,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

    program = build_decode_projection_program(
        batch=batch,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    tensor_specs = build_tensor_specs(
        batch=batch,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_decode_projection,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=2e-2,
            atol=2e-2,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=backend,
            enable_profiling=enable_profiling,
        ),
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = compile_and_run(
        platform=args.platform,
        device_id=args.device,
        enable_profiling=args.enable_profiling,
    )
    if not result.passed:
        if result.error:
            print(f"Result: {result.error}")
        raise SystemExit(1)
