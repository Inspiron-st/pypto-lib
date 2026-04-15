# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek V3.2-EXP prefill BACK Scope 2 — post RMSNorm.

This scope extracts the post-RMSNorm from deepSeek V3.2 prefill_back.
Computation flow matches prefill_back.py lines 108-125:
  1. sq_sum = sum(x^2) over hidden dimension
  2. inv_rms = rsqrt(sq_sum / hidden_size + eps)
  3. normed = x * inv_rms * gamma
"""
from __future__ import annotations

import pypto.language as pl


BATCH = 16
MAX_SEQ = 4096
HIDDEN = 7168

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

K_CHUNK = 512
TOK_TILE = 8


def build_deepseek_v3_2_prefill_back_scope2_program(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
):
    BATCH_CFG = batch
    MAX_SEQ_CFG = max_seq_len
    HIDDEN_CFG = hidden_size

    HIDDEN_INV_CFG = 1.0 / HIDDEN_CFG

    HIDDEN_BLOCKS = (HIDDEN_CFG + K_CHUNK - 1) // K_CHUNK

    @pl.program
    class DeepSeekV32PrefillBackScope2:
        @pl.function(type=pl.FunctionType.Opaque)
        def scope2(
            self,
            resid1: pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.FP32],
            seq_lens: pl.Tensor[[BATCH_CFG], pl.INT32],
            post_rms_weight: pl.Tensor[[1, HIDDEN_CFG], pl.FP32],
            post_norm: pl.Out[pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16]],
        ) -> pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16]:
            for b in pl.parallel(0, BATCH_CFG, 1):
                seq_len_b = pl.tensor.read(seq_lens, [b])
                tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                for p0_idx in pl.range(tok_blocks):
                    p0 = p0_idx * TOK_TILE
                    valid_tok = pl.min(TOK_TILE, seq_len_b - p0)

                    with pl.incore():
                        sq_sum = pl.create_tensor([TOK_TILE, 1], dtype=pl.FP32)
                        sq_sum = pl.mul(sq_sum, 0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk_raw = pl.reshape(
                                pl.slice(resid1, [1, TOK_TILE, K_CHUNK], [b, p0, k0],
                                        valid_shape=[1, valid_tok, K_CHUNK]),
                                [TOK_TILE, K_CHUNK]
                            )
                            x_chunk = pl.fillpad(x_chunk_raw, pad_value=pl.PadValue.zero)
                            sq_sum = pl.add(sq_sum, pl.row_sum(pl.mul(x_chunk, x_chunk)))

                        inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV_CFG), EPS))

                    post_norm_tile = pl.create_tensor([TOK_TILE, HIDDEN_CFG], dtype=pl.BF16)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        with pl.incore():
                            x_chunk_raw = pl.reshape(
                                pl.slice(resid1, [1, TOK_TILE, K_CHUNK], [b, p0, k0],
                                        valid_shape=[1, valid_tok, K_CHUNK]),
                                [TOK_TILE, K_CHUNK]
                            )
                            x_chunk = pl.fillpad(x_chunk_raw, pad_value=pl.PadValue.zero)
                            gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                            normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                            post_norm_tile = pl.assemble(post_norm_tile, pl.cast(normed, target_type=pl.BF16), [0, k0])

                    post_norm = pl.assemble(post_norm, post_norm_tile, [b, p0, 0])

            return post_norm

    return DeepSeekV32PrefillBackScope2


def build_tensor_specs(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
):
    import torch
    from pypto.runtime import TensorSpec

    seq_lens_data = torch.randint(1, max_seq_len + 1, (batch,), dtype=torch.int32)

    return [
        TensorSpec("resid1", [batch, max_seq_len, hidden_size], torch.float32, init_value=torch.randn),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=seq_lens_data),
        TensorSpec("post_rms_weight", [1, hidden_size], torch.float32, init_value=torch.randn),
        TensorSpec("post_norm", [batch, max_seq_len, hidden_size], torch.bfloat16, is_output=True),
    ]


def golden_scope2(tensors, params):
    """PyTorch reference for Scope 2: post RMSNorm."""
    import torch

    resid1 = tensors["resid1"]
    seq_lens = tensors["seq_lens"]
    post_rms_weight = tensors["post_rms_weight"]

    batch = resid1.shape[0]
    max_seq = resid1.shape[1]
    hidden_size = resid1.shape[2]
    eps = 1e-6

    post_norm = torch.zeros(batch, max_seq, hidden_size, dtype=torch.bfloat16)

    for b in range(batch):
        seq_len_b = seq_lens[b].item()
        for p0 in range(0, seq_len_b, TOK_TILE):
            valid_tok = min(TOK_TILE, seq_len_b - p0)
            resid1_tile_full = resid1[b, p0:p0 + TOK_TILE, :]
            resid1_tile_full[valid_tok:, :] = 0.0
            sq_sum = (resid1_tile_full ** 2).sum(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(sq_sum / hidden_size + eps)
            normed = resid1_tile_full * inv_rms * post_rms_weight.float()
            post_norm[b, p0:p0 + TOK_TILE, :] = normed.bfloat16()

    tensors["post_norm"][:] = post_norm


def compile_and_run(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    platform: str = "a2a3",
    device_id: int = 0,
    dump_passes: bool = True,
    runtime_profiling: bool = False,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

    program = build_deepseek_v3_2_prefill_back_scope2_program(
        batch=batch,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
    )
    tensor_specs = build_tensor_specs(
        batch=batch,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_scope2,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=2e-2,
            atol=2e-2,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=backend,
            runtime_profiling=runtime_profiling,
        ),
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--max-seq", type=int, default=MAX_SEQ)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    args = parser.parse_args()

    result = compile_and_run(
        batch=args.batch,
        max_seq_len=args.max_seq,
        hidden_size=args.hidden,
        platform=args.platform,
        device_id=args.device,
        runtime_profiling=args.runtime_profiling,
    )
    if not result.passed:
        if result.error:
            print(f"Result: {result.error}")
        raise SystemExit(1)