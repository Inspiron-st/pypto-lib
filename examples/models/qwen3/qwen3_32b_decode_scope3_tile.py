# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Qwen3 Scope 3 tile/helper version: output projection + residual + post RMSNorm + MLP + residual.

This mirrors `qwen3_32b_decode_scope3.py`, but keeps the fused gate/up and
down-projection paths as explicit InCore helpers so the optimized tile-level
structure is easy to inspect in one file.
"""

import pypto.language as pl

BATCH = 16
HIDDEN = 8192
INTERMEDIATE = 25600

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

K_CHUNK = 128
Q_OUT_CHUNK = 64
MLP_OUT_CHUNK = 64
BATCH_TILE = 16


def build_qwen3_scope3_tile_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    BATCH_CFG = batch
    HIDDEN_CFG = hidden_size
    INTER_CFG = intermediate_size

    if HIDDEN_CFG % (2 * K_CHUNK) != 0:
        raise ValueError("scope3 fused down projection requires hidden_size divisible by 256")

    HIDDEN_BLOCKS = HIDDEN_CFG // K_CHUNK
    Q_OUT_BLOCKS = HIDDEN_CFG // Q_OUT_CHUNK
    MLP_OUT_BLOCKS = INTER_CFG // MLP_OUT_CHUNK
    HIDDEN_OUT_PAIR_BLOCKS = HIDDEN_BLOCKS // 2

    @pl.program
    class Qwen3Scope3Tile:
        @pl.function(type=pl.FunctionType.InCore)
        def fused_gate_up_reduce(
            self,
            post_norm_tile: pl.Tensor[[BATCH_TILE, HIDDEN_CFG], pl.BF16],
            w_gate: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_up: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            out_col: pl.Scalar[pl.INDEX],
            gate_out: pl.Out[pl.Tensor[[BATCH_TILE, MLP_OUT_CHUNK], pl.FP32]],
            up_out: pl.Out[pl.Tensor[[BATCH_TILE, MLP_OUT_CHUNK], pl.FP32]],
        ) -> tuple[
            pl.Tensor[[BATCH_TILE, MLP_OUT_CHUNK], pl.FP32],
            pl.Tensor[[BATCH_TILE, MLP_OUT_CHUNK], pl.FP32],
        ]:
            # InCore stage: fused gate/up projection for one MLP output tile.
            # Load one post-norm activation chunk, then accumulate both
            # gate and up outputs across all hidden chunks.
            post_l1 = pl.load(
                post_norm_tile,
                [0, 0],
                [BATCH_TILE, K_CHUNK],
                target_memory=pl.MemorySpace.Mat,
            )
            gate_w_l1 = pl.load(
                w_gate,
                [0, out_col],
                [K_CHUNK, MLP_OUT_CHUNK],
                target_memory=pl.MemorySpace.Mat,
            )
            up_w_l1 = pl.load(
                w_up,
                [0, out_col],
                [K_CHUNK, MLP_OUT_CHUNK],
                target_memory=pl.MemorySpace.Mat,
            )
            post_l0 = pl.move(post_l1, target_memory=pl.MemorySpace.Left)
            gate_w_l0 = pl.move(gate_w_l1, target_memory=pl.MemorySpace.Right)
            up_w_l0 = pl.move(up_w_l1, target_memory=pl.MemorySpace.Right)
            gate_acc = pl.matmul(post_l0, gate_w_l0)
            up_acc = pl.matmul(post_l0, up_w_l0)

            for kb in pl.range(1, HIDDEN_BLOCKS):
                k0 = kb * K_CHUNK
                post_l1_i = pl.load(
                    post_norm_tile,
                    [0, k0],
                    [BATCH_TILE, K_CHUNK],
                    target_memory=pl.MemorySpace.Mat,
                )
                gate_w_l1_i = pl.load(
                    w_gate,
                    [k0, out_col],
                    [K_CHUNK, MLP_OUT_CHUNK],
                    target_memory=pl.MemorySpace.Mat,
                )
                up_w_l1_i = pl.load(
                    w_up,
                    [k0, out_col],
                    [K_CHUNK, MLP_OUT_CHUNK],
                    target_memory=pl.MemorySpace.Mat,
                )
                post_l0_i = pl.move(post_l1_i, target_memory=pl.MemorySpace.Left)
                gate_w_l0_i = pl.move(gate_w_l1_i, target_memory=pl.MemorySpace.Right)
                up_w_l0_i = pl.move(up_w_l1_i, target_memory=pl.MemorySpace.Right)
                gate_acc = pl.matmul_acc(gate_acc, post_l0_i, gate_w_l0_i)
                up_acc = pl.matmul_acc(up_acc, post_l0_i, up_w_l0_i)

            gate_out = pl.store(gate_acc, [0, 0], gate_out)
            up_out = pl.store(up_acc, [0, 0], up_out)
            return gate_out, up_out

        @pl.function(type=pl.FunctionType.InCore)
        def fused_down_reduce(
            self,
            mlp_tile: pl.Tensor[[BATCH_TILE, INTER_CFG], pl.BF16],
            w_down: pl.Tensor[[INTER_CFG, HIDDEN_CFG], pl.BF16],
            out_col0: pl.Scalar[pl.INDEX],
            out_col1: pl.Scalar[pl.INDEX],
            down_out0: pl.Out[pl.Tensor[[BATCH_TILE, K_CHUNK], pl.FP32]],
            down_out1: pl.Out[pl.Tensor[[BATCH_TILE, K_CHUNK], pl.FP32]],
        ) -> tuple[
            pl.Tensor[[BATCH_TILE, K_CHUNK], pl.FP32],
            pl.Tensor[[BATCH_TILE, K_CHUNK], pl.FP32],
        ]:
            # InCore stage: fused down projection for two adjacent hidden tiles.
            # Reuse the same MLP activation chunk stream while accumulating two
            # output columns in parallel.
            mlp_l1 = pl.load(
                mlp_tile,
                [0, 0],
                [BATCH_TILE, MLP_OUT_CHUNK],
                target_memory=pl.MemorySpace.Mat,
            )
            w_down_l1_0 = pl.load(
                w_down,
                [0, out_col0],
                [MLP_OUT_CHUNK, K_CHUNK],
                target_memory=pl.MemorySpace.Mat,
            )
            w_down_l1_1 = pl.load(
                w_down,
                [0, out_col1],
                [MLP_OUT_CHUNK, K_CHUNK],
                target_memory=pl.MemorySpace.Mat,
            )
            mlp_l0 = pl.move(mlp_l1, target_memory=pl.MemorySpace.Left)
            w_down_l0_0 = pl.move(w_down_l1_0, target_memory=pl.MemorySpace.Right)
            w_down_l0_1 = pl.move(w_down_l1_1, target_memory=pl.MemorySpace.Right)
            down_acc0 = pl.matmul(mlp_l0, w_down_l0_0)
            down_acc1 = pl.matmul(mlp_l0, w_down_l0_1)

            for ob in pl.range(1, MLP_OUT_BLOCKS):
                o0 = ob * MLP_OUT_CHUNK
                mlp_l1_i = pl.load(
                    mlp_tile,
                    [0, o0],
                    [BATCH_TILE, MLP_OUT_CHUNK],
                    target_memory=pl.MemorySpace.Mat,
                )
                w_down_l1_0_i = pl.load(
                    w_down,
                    [o0, out_col0],
                    [MLP_OUT_CHUNK, K_CHUNK],
                    target_memory=pl.MemorySpace.Mat,
                )
                w_down_l1_1_i = pl.load(
                    w_down,
                    [o0, out_col1],
                    [MLP_OUT_CHUNK, K_CHUNK],
                    target_memory=pl.MemorySpace.Mat,
                )
                mlp_l0_i = pl.move(mlp_l1_i, target_memory=pl.MemorySpace.Left)
                w_down_l0_0_i = pl.move(w_down_l1_0_i, target_memory=pl.MemorySpace.Right)
                w_down_l0_1_i = pl.move(w_down_l1_1_i, target_memory=pl.MemorySpace.Right)
                down_acc0 = pl.matmul_acc(down_acc0, mlp_l0_i, w_down_l0_0_i)
                down_acc1 = pl.matmul_acc(down_acc1, mlp_l0_i, w_down_l0_1_i)

            down_out0 = pl.store(down_acc0, [0, 0], down_out0)
            down_out1 = pl.store(down_acc1, [0, 0], down_out1)
            return down_out0, down_out1

        @pl.function(type=pl.FunctionType.InCore)
        def fused_down_writeback(
            self,
            down_acc0: pl.Tensor[[BATCH_TILE, K_CHUNK], pl.FP32],
            down_acc1: pl.Tensor[[BATCH_TILE, K_CHUNK], pl.FP32],
            resid1_tile: pl.Tensor[[BATCH_TILE, HIDDEN_CFG], pl.FP32],
            out_row: pl.Scalar[pl.INDEX],
            out_col0: pl.Scalar[pl.INDEX],
            out_col1: pl.Scalar[pl.INDEX],
            output: pl.Out[pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16]],
        ) -> pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16]:
            # InCore stage: final residual add + BF16 writeback for the same
            # two hidden tiles produced by fused_down_reduce.
            out_chunk0 = pl.add(
                down_acc0,
                pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, out_col0]),
            )
            out_chunk1 = pl.add(
                down_acc1,
                pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, out_col1]),
            )
            output = pl.assemble(output, pl.cast(out_chunk0, target_type=pl.BF16), [out_row, out_col0])
            output = pl.assemble(output, pl.cast(out_chunk1, target_type=pl.BF16), [out_row, out_col1])
            return output

        @pl.function(type=pl.FunctionType.Opaque)
        def scope3(
            self,
            attn_out: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
            hidden_states: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
            wo: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.BF16],
            post_rms_weight: pl.Tensor[[1, HIDDEN_CFG], pl.FP32],
            w_gate: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_up: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_down: pl.Tensor[[INTER_CFG, HIDDEN_CFG], pl.BF16],
            out: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
        ) -> pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16]:
            for b0 in pl.range(0, BATCH_CFG, BATCH_TILE):
                resid1_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.FP32)

                # Stage 0 InCore: output projection over Q_OUT_CHUNK tiles.
                for ob in pl.range(Q_OUT_BLOCKS):
                    o0 = ob * Q_OUT_CHUNK

                    with pl.incore():
                        a_chunk_0 = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, 0])
                        w_chunk_0 = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [0, o0])
                        o_acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)
                        for kb in pl.range(1, HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            a_chunk = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, k0])
                            w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                            o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)

                    # Stage 1 InCore: add the residual tile into the projected output.
                    with pl.incore():
                        resid = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, Q_OUT_CHUNK], [b0, o0]),
                            target_type=pl.FP32,
                        )
                        resid_sum = pl.add(o_acc, resid)
                        resid1_tile = pl.assemble(resid1_tile, resid_sum, [0, o0])

                # Stage 2 InCore: post-attention RMSNorm.
                # First accumulate row-wise squared sums, then apply inv_rms and gamma.
                post_norm_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.BF16)
                with pl.incore():
                    sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        sq_sum = pl.add(
                            sq_sum,
                            pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH_TILE]),
                        )
                    inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))

                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(
                            pl.row_expand_mul(x_chunk, pl.reshape(inv_rms, [BATCH_TILE, 1])),
                            gamma,
                        )
                        normed_bf16 = pl.cast(normed, target_type=pl.BF16)
                        post_norm_tile = pl.assemble(post_norm_tile, normed_bf16, [0, k0])

                # Stage 3 InCore helper: fused gate/up projection per MLP output tile.
                # Stage 4 InCore: SiLU and gating, then materialize the MLP tile.
                mlp_tile = pl.create_tensor([BATCH_TILE, INTER_CFG], dtype=pl.BF16)
                for ob in pl.range(MLP_OUT_BLOCKS):
                    o0 = ob * MLP_OUT_CHUNK
                    gate_acc = pl.create_tensor([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                    up_acc = pl.create_tensor([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                    gate_acc, up_acc = self.fused_gate_up_reduce(
                        post_norm_tile,
                        w_gate,
                        w_up,
                        o0,
                        gate_acc,
                        up_acc,
                    )

                    with pl.auto_incore():
                        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                        mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                        mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
                        mlp_tile = pl.assemble(mlp_tile, mlp_chunk_bf16, [0, o0])

                # Stage 5 InCore helper: fused down projection for two hidden tiles.
                # Stage 6 InCore helper: residual add + final BF16 writeback.
                for dob in pl.range(HIDDEN_OUT_PAIR_BLOCKS):
                    d0 = dob * 2 * K_CHUNK
                    d1 = d0 + K_CHUNK
                    down_acc0 = pl.create_tensor([BATCH_TILE, K_CHUNK], dtype=pl.FP32)
                    down_acc1 = pl.create_tensor([BATCH_TILE, K_CHUNK], dtype=pl.FP32)
                    down_acc0, down_acc1 = self.fused_down_reduce(
                        mlp_tile,
                        w_down,
                        d0,
                        d1,
                        down_acc0,
                        down_acc1,
                    )
                    out = self.fused_down_writeback(
                        down_acc0,
                        down_acc1,
                        resid1_tile,
                        b0,
                        d0,
                        d1,
                        out,
                    )

            return out

    return Qwen3Scope3Tile


def golden(tensors: dict, params: dict | None = None) -> None:
    """Reference computation for Scope 3 tile/helper version."""
    import torch

    attn_out = tensors["attn_out"]
    hidden_states = tensors["hidden_states"]
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]

    eps = 1e-6

    o_proj = torch.matmul(attn_out.float(), wo.float())
    resid1 = o_proj + hidden_states.float()

    variance = resid1.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    normed_bf16 = (resid1 * inv_rms * post_rms_weight).bfloat16()

    gate = torch.matmul(normed_bf16.float(), w_gate.float())
    up = torch.matmul(normed_bf16.float(), w_up.float())
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    down = torch.matmul(mlp_bf16.float(), w_down.float())

    tensors["out"][:] = (down + resid1).bfloat16()


def build_tensor_specs(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    import torch
    from pypto.runtime import TensorSpec

    def init_attn_out():
        return torch.rand(batch, hidden_size) - 0.5

    def init_hidden_states():
        return torch.rand(batch, hidden_size) - 0.5

    def init_wo():
        return (torch.rand(hidden_size, hidden_size) - 0.5) / hidden_size ** 0.5

    def init_post_rms_weight():
        return torch.ones(1, hidden_size)

    def init_w_gate():
        return (torch.rand(hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5

    def init_w_up():
        return (torch.rand(hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5

    def init_w_down():
        return (torch.rand(intermediate_size, hidden_size) - 0.5) / intermediate_size ** 0.5

    return [
        TensorSpec("attn_out", [batch, hidden_size], torch.bfloat16, init_value=init_attn_out),
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("wo", [hidden_size, hidden_size], torch.bfloat16, init_value=init_wo),
        TensorSpec("post_rms_weight", [1, hidden_size], torch.float32, init_value=init_post_rms_weight),
        TensorSpec("w_gate", [hidden_size, intermediate_size], torch.bfloat16, init_value=init_w_gate),
        TensorSpec("w_up", [hidden_size, intermediate_size], torch.bfloat16, init_value=init_w_up),
        TensorSpec("w_down", [intermediate_size, hidden_size], torch.bfloat16, init_value=init_w_down),
        TensorSpec("out", [batch, hidden_size], torch.bfloat16, is_output=True),
    ]


def compile_and_run(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    platform: str = "a5",
    device_id: int = 0,
    work_dir: str | None = None,
    dump_passes: bool = True,
    runtime_profiling: bool = False,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

    program = build_qwen3_scope3_tile_program(
        batch=batch,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    tensor_specs = build_tensor_specs(
        batch=batch,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=3e-3,
            atol=3e-3,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=backend,
            runtime_profiling=runtime_profiling,
        ),
    )
    if not result.passed and result.error and "code_runner" in result.error:
        print("Result: COMPILE OK — device run skipped (code_runner not found).")
    if not result.passed and result.error:
        print(f"Result: {result.error}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = compile_and_run(
        platform=args.platform,
        device_id=args.device,
        runtime_profiling=args.runtime_profiling,
    )
    if not result.passed:
        if result.error:
            print(f"Result: {result.error}")
        raise SystemExit(1)
