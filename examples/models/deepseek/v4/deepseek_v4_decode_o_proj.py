# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek-V4 grouped output projection (fused).

Corresponds to model.py Attention.forward lines 537-542:
    o = o.view(bsz, seqlen, n_local_groups, -1)
    wo_a = self.wo_a.weight.view(n_local_groups, o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    x = self.wo_b(o.flatten(2))

Stage A (grouped einsum over the local groups, ``wo_a``) and Stage B
(``wo_b`` GEMM that reduces across groups) are fused into one program:
Stage A's BF16 output ``o_r`` lives entirely in on-chip memory between
the two stages and is reused as Stage B's left operand. The intermediate
``o_r`` corresponds to the local Python tensor ``o`` produced on
model.py line 540 by ``torch.einsum("bsgd,grd->bsgr", ...)``.

Tensor layouts on the program boundary:
- ``o``    : ``[T, H, HEAD_DIM]``         BF16 -- verbatim attention output
             from ``deepseek_v4_decode_single_layer.md``. The kernel packs
             this internally into a group-major workspace; see prologue
             below for the rationale (workaround for
             hw-native-sys/pypto#1212).
- ``wo_a`` : ``[O_GROUPS, O_LORA, O_GROUP_IN]`` BF16 -- matches the
             upstream ``self.wo_a.weight.view(G, R, O_GROUP_IN)``
             contract. The kernel reshapes this internally to
             ``[G*R, O_GROUP_IN]`` before Stage A so the cube matmul does
             not consume a 3D leading-dim-strided RHS operand (workaround
             for hw-native-sys/pypto#1212).
- ``wo_b`` : ``[D, O_GROUPS * O_LORA]``   BF16 -- matches the upstream
             ``RowParallelLinear`` weight verbatim; consumed via
             ``b_trans=True``.
- ``attn_out`` : ``[T, D]``                BF16 -- flattened token view of
             the upstream ``[B, S, D]`` output with ``T = B*S``. This keeps
             the o_proj kernel boundary consistent with the surrounding
             token-major pypto examples; downstream consumers may view it as
             ``[B, S, D]`` when needed.

Rationale for the prologue: indexing ``o`` directly as
``[T, G*O_GROUP_IN]`` with a column offset, or 3D-slicing
``[T, hpg, HEAD_DIM]`` and reshape, both yield a per-token-strided LHS
operand that mis-compiles on a2a3 (only row 0 of the resulting matmul
comes back correct -- see hw-native-sys/pypto#1212). The prologue
materialises a contiguous group-major workspace so Stage A operates on
a plain ``[T, O_GROUP_IN]`` tile per group. Once #1212 is fixed the
prologue can be replaced by an in-place 3D slice + reshape view.

``wo_a`` uses the same defensive pattern. The mathematically direct
Stage-A RHS slice is ``wo_a[g, r0:r1, d0:d1]`` consumed with
``b_trans=True``; on a2a3 this exercises the RHS side of #1212. The
kernel therefore first views ``wo_a`` as a contiguous 2D
``[G*R, O_GROUP_IN]`` tensor and slices ``[R_CHUNK, K_CHUNK]`` panels
from that view. TODO(#1212): remove this reshape workaround once grouped
strided RHS matmul is fixed.

FP32 accumulator is mandatory throughout: the ISA's TileType::Acc only
supports ``float`` or ``int32_t`` (pto-isa: tmatmul.md).
"""


import pypto.language as pl


B          = 16                 # demo 4
S          = 1
T          = B * S
D          = 4096               # v4-pro 7168
H          = 64                 # v4-pro 128
HEAD_DIM   = 512
O_LORA     = 1024
O_GROUPS   = 8                  # v4-pro 16
HEADS_PER_GROUP = H // O_GROUPS          # 8 heads per group
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM  # 4096 (matches v4-pro by coincidence)
GR         = O_GROUPS * O_LORA           # 8192 (v4-pro 16384)

# Stage A tile shape (M=T; K over O_GROUP_IN; N over O_LORA).
A_K_CHUNK  = 128
A_N_CHUNK  = 128

# Stage B tile shape (M=T; K over GR; N over D).
B_K_CHUNK  = 128
B_N_CHUNK  = 256


def build_deepseek_v4_decode_o_proj_program():
    assert H % O_GROUPS == 0, (
        f"H ({H}) must be divisible by O_GROUPS ({O_GROUPS})"
    )
    assert O_GROUP_IN % A_K_CHUNK == 0, (
        f"O_GROUP_IN ({O_GROUP_IN}) must be divisible by A_K_CHUNK ({A_K_CHUNK})"
    )
    assert O_LORA % A_N_CHUNK == 0, (
        f"O_LORA ({O_LORA}) must be divisible by A_N_CHUNK ({A_N_CHUNK})"
    )
    assert GR % B_K_CHUNK == 0, (
        f"GR ({GR}) must be divisible by B_K_CHUNK ({B_K_CHUNK})"
    )
    assert D % B_N_CHUNK == 0, (
        f"D ({D}) must be divisible by B_N_CHUNK ({B_N_CHUNK})"
    )

    A_K_BLOCKS = O_GROUP_IN // A_K_CHUNK
    A_N_BLOCKS = O_LORA // A_N_CHUNK
    B_K_BLOCKS = GR // B_K_CHUNK
    B_N_BLOCKS = D // B_N_CHUNK

    @pl.program
    class DeepSeekV4DecodeOProj:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_o_proj(
            self,
            o:        pl.Tensor[[T, H, HEAD_DIM],                     pl.BF16],
            wo_a:     pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN],       pl.BF16],
            wo_b:     pl.Tensor[[D, GR],                              pl.BF16],
            attn_out: pl.Out[pl.Tensor[[T, D],                        pl.BF16]],
        ):
            # Group-major packed view of ``o``. See module docstring for the
            # rationale (workaround for hw-native-sys/pypto#1212).
            o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)

            # 2D view of upstream wo_a[g, r, d] as wo_a_2d[g*R + r, d].
            # TODO(#1212): use the direct 3D slice once a2a3 handles grouped
            # strided RHS matmul correctly.
            wo_a_2d = pl.reshape(wo_a, [GR, O_GROUP_IN])

            # Internal workspace bridging Stage A and Stage B. Mirrors the
            # local Python tensor ``o_r`` produced on model.py line 540 by
            # ``torch.einsum("bsgd,grd->bsgr", ...)``.
            o_r = pl.create_tensor([T, GR], dtype=pl.BF16)

            # ---- Prologue: pack o[t, g*hpg:(g+1)*hpg, :] -> o_packed[g*T+t, :] ----
            # Each per-token slice [hpg, HEAD_DIM] = O_GROUP_IN BF16 is contiguous
            # in the source, so the copy is a single DMA per (g, t).
            for g in pl.range(O_GROUPS):
                head_off = g * HEADS_PER_GROUP
                row_base = g * T
                for t in pl.range(T):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        src_3d = pl.slice(
                            o, [1, HEADS_PER_GROUP, HEAD_DIM], [t, head_off, 0]
                        )
                        src_2d = pl.reshape(src_3d, [1, O_GROUP_IN])
                        o_packed = pl.assemble(o_packed, src_2d, [row_base + t, 0])

            # ---- Stage A: o_r[t, g, r] = sum_d o_packed[g*T+t, d] * wo_a[g, d, r] ----
            for g in pl.range(O_GROUPS):
                row_base_o = g * T
                w_row_g    = g * O_LORA
                out_col_g  = g * O_LORA

                for nb in pl.range(A_N_BLOCKS):
                    n0 = nb * A_N_CHUNK

                    with pl.at(level=pl.Level.CORE_GROUP):
                        xa0 = pl.slice(o_packed, [T, A_K_CHUNK],         [row_base_o, 0])
                        wa0 = pl.slice(wo_a_2d,  [A_N_CHUNK, A_K_CHUNK], [w_row_g + n0, 0])
                        acc_a = pl.matmul(xa0, wa0, b_trans=True, out_dtype=pl.FP32)

                        for kb in pl.range(1, A_K_BLOCKS):
                            k0 = kb * A_K_CHUNK
                            xa_chunk = pl.slice(
                                o_packed, [T, A_K_CHUNK], [row_base_o, k0]
                            )
                            wa_chunk = pl.slice(
                                wo_a_2d, [A_N_CHUNK, A_K_CHUNK], [w_row_g + n0, k0]
                            )
                            acc_a = pl.matmul_acc(acc_a, xa_chunk, wa_chunk, b_trans=True)

                    # Separate CORE_GROUP scope so the BF16 cast lands on the
                    # vector path and the GM store sees a row-major tile (the
                    # matmul accumulator is column-major in the cube buffer).
                    with pl.at(level=pl.Level.CORE_GROUP):
                        o_r = pl.assemble(
                            o_r,
                            pl.cast(acc_a, target_type=pl.BF16),
                            [0, out_col_g + n0],
                        )

            # ---- Stage B: attn_out = o_r @ wo_b^T  ----
            for nb in pl.range(B_N_BLOCKS):
                n0 = nb * B_N_CHUNK

                with pl.at(level=pl.Level.CORE_GROUP):
                    xb0 = pl.slice(o_r,  [T, B_K_CHUNK],          [0, 0])
                    wb0 = pl.slice(wo_b, [B_N_CHUNK, B_K_CHUNK],  [n0, 0])
                    acc_b = pl.matmul(xb0, wb0, b_trans=True, out_dtype=pl.FP32)

                    for kb in pl.range(1, B_K_BLOCKS):
                        k0 = kb * B_K_CHUNK
                        xb_chunk = pl.slice(o_r,  [T, B_K_CHUNK],         [0, k0])
                        wb_chunk = pl.slice(wo_b, [B_N_CHUNK, B_K_CHUNK], [n0, k0])
                        acc_b = pl.matmul_acc(acc_b, xb_chunk, wb_chunk, b_trans=True)

                # Separate CORE_GROUP scope (see Stage A note above).
                with pl.at(level=pl.Level.CORE_GROUP):
                    attn_out = pl.assemble(
                        attn_out,
                        pl.cast(acc_b, target_type=pl.BF16),
                        [0, n0],
                    )

            return attn_out

    return DeepSeekV4DecodeOProj


def golden_deepseek_v4_decode_o_proj(tensors):
    """Torch reference for model.py Attention.forward L537-541.

    Inputs:
      - ``o``    : ``[T, H, HEAD_DIM]``                (verbatim attention output)
      - ``wo_a`` : ``[O_GROUPS, O_LORA, O_GROUP_IN]`` (upstream layout)
      - ``wo_b`` : ``[D, O_GROUPS * O_LORA]``

    The intermediate ``o_r`` is cast through BF16 to mirror the kernel's
    on-chip workspace dtype before being multiplied by ``wo_b``.
    """
    import torch

    o    = tensors["o"].float()                            # [T, H, HEAD_DIM]
    wo_a = tensors["wo_a"].float()                         # [G, O_LORA, O_GROUP_IN]
    wo_b = tensors["wo_b"].float()                         # [D, G*O_LORA]

    # Reconstruct the upstream einsum layouts:
    #   o    -> [B, S, G, O_GROUP_IN]    (verbatim view of [T, H, HEAD_DIM])
    #   wo_a -> [G, R, O_GROUP_IN]
    o_model    = o.view(B, S, O_GROUPS, O_GROUP_IN)
    wo_a_model = wo_a                                      # [G, R, O_GROUP_IN]

    # Stage A: upstream einsum verbatim.
    o_r = torch.einsum("bsgd,grd->bsgr", o_model, wo_a_model)  # [B, S, G, R]
    # Match kernel's on-chip BF16 workspace precision before Stage B.
    o_r = o_r.to(torch.bfloat16).float()
    # Stage B: wo_b expansion back to model dim.
    out = o_r.flatten(2).view(T, GR) @ wo_b.T                  # [T, D]

    tensors["attn_out"][:] = out.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_o():
        return torch.randn(T, H, HEAD_DIM) * 0.05
    def init_wo_a():
        return torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / (O_GROUP_IN ** 0.5)
    def init_wo_b():
        return torch.randn(D, GR) / (GR ** 0.5)

    return [
        TensorSpec("o",        [T, H, HEAD_DIM],                    torch.bfloat16, init_value=init_o),
        TensorSpec("wo_a",     [O_GROUPS, O_LORA, O_GROUP_IN],      torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b",     [D, GR],                             torch.bfloat16, init_value=init_wo_b),
        TensorSpec("attn_out", [T, D],                              torch.bfloat16, is_output=True),
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
        program=build_deepseek_v4_decode_o_proj_program(),
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_o_proj,
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
