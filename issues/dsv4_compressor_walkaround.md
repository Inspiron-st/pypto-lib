# DeepSeek-V4 Compressor — pypto syntax walkarounds

Workarounds discovered while building `deepseek_v4_decode_compressor_draft.py` (CV-split mode). Each entry: symptom → root cause → workaround.

## 1. 3D `pl.slice` / `pl.assemble` on stateful (InOut) tensors triggers ND/NZ layout mismatch

**Symptom**: ptoas `error: layout mismatch: user-specified layout=nd but inferred=nz` on `make_tensor_view` lines for 3D state tensors after a vector pass reads them.

**Root cause**: When a vector kernel uses 3D `partition_view` over a GM tensor that participates in cube-output writes upstream, ptoas infers NZ layout for the view while the frontend emits ND. Specifically, the [B, STATE_LEN, OUT_DIM] InOut tensor written by `state_scatter` (downstream of `pl.matmul`) cannot be read back via 3D slice in another vector block.

**Workaround**: Stay 2D for any GM tensor that is both written and read by vector ops. Reshape 3D parameters to 2D at function entry, do all vector slice/assemble in 2D, reshape back to 3D only as the final SSA step before the InOut shape is consumed by the runtime. Mirrors `deepseek_v4_decode_hc_pre.py`'s pattern.

```python
# Avoid 3D slice/assemble on stateful tensors:
state_2d = pl.reshape(state_3d, [B * STATE_LEN, OUT_DIM])
with pl.at(...):
    chunk = pl.slice(state_2d, [...], [...])
    state_2d = pl.assemble(state_2d, chunk, [...])
state_3d_back = pl.reshape(state_2d, [B, STATE_LEN, OUT_DIM])
```

Setting `pl.matmul(..., c_matrix_nz=False)` does **not** fix this — the layout error originates from the downstream view, not the cube output.

## 2. `pl.col_expand_add` does not exist; use `add(matrix, col_expand_mul(ones, row_vec))`

**Symptom**: No documented `col_expand_add` op for broadcasting a `[1, N]` row vector into each row of an `[M, N]` tile.

**Workaround** (from `deepseek_v4_decode_hc_pre.py:109-112`):

```python
ones = pl.full([M, N], dtype=pl.FP32, value=1.0)
broadcast = pl.col_expand_mul(ones, row_vec)  # row_vec: [1, N]
result = pl.add(matrix, broadcast)
```

## 3. Tensor-level `pl.col_max` / `pl.col_sum` are not exposed; only `row_*` reductions

**Symptom**: `Error: Unknown tensor operation: col_max`. The frontend coding-style doc lists `block.col_max`/`block.col_sum` in the IR registry, but they're not exposed via the unified `pl.*` dispatch (only `pl.row_max` / `pl.row_sum` are reachable from a `pl.slice` result).

**Workaround**: Reduce manually with `pl.maximum` / `pl.add` over single-row slices indexed by a `pl.range` loop. Keep the loop-carried accumulator as a regular Python variable updated each iteration — pypto's SSA tracks it the same way `mix_col` is updated in `deepseek_v4_decode_hc_pre.py:81-91`.

```python
s_max = pl.slice(scs, [1, HEAD_CHUNK], [row_b, h0])
for s in pl.range(1, STATE_LEN):
    s_row = pl.slice(scs, [1, HEAD_CHUNK], [row_b + s, h0])
    s_max = pl.maximum(s_max, s_row)
```

`pl.transpose` + `pl.row_max` was tried first but failed in the optimizer with `'pto.alloc_tile' op valid_row operand is required because result type v_row is ?` — transposing a `pl.slice` result into the column-becoming-row direction loses the static row count needed for tile allocation.

## 5. Manual softmax+pool over STATE_LEN produces ~9% over-magnification (UNRESOLVED)

**Symptom**: With `score_state` initialized to `-inf` (only slot 7 has finite scores after `block 3`), the manual softmax+weighted-sum in `block 5+6` should mathematically reduce to `pooled = kvs[slot 7]`. Empirically the kernel produces values ~9% larger (kernel/golden ratio 1.05-1.09 across all elements).

**AB-tests confirm**:
- Override `pooled = kv_fp32[:, HEAD_DIM:OUT_DIM]` (skip block 5+6) → full pipeline passes (206 mismatches, all 1-ULP BF16 noise on RoPE).
- Replace block 5+6 with hardcoded `pooled = kvs[slot 7]` → 189 mismatches (same noise level).
- Block 1 matmul, block 4 gather, block 8 RMSNorm, block 9 RoPE, block 11 cast — all correct in isolation.
- Init = `float("-inf")`, `-1e20`, or `-100.0` all give the same ~9% bias → not an `exp(-inf)` saturation issue.
- `pl.col_max` / `pl.col_sum` aren't exposed; `pl.transpose` on sliced tiles loses static row count → only viable form is the manual reduction.

**Where**: `examples/models/deepseek/v4/deepseek_v4_decode_compressor_draft.py:softmax_pool` block.

**Status**: Unresolved. Worth filing as a pypto issue (with a minimal repro) once the cause is narrowed further. Likely candidates: tile reuse across `pl.range` iterations corrupting `s_max` / `e_sum` / `pooled_acc`, or a hardware quirk in `pl.exp` near 0/-inf.

---

## 4. Pure Python `for` and list comprehensions inside `pl.at` blocks are not allowed

**Symptom**:
- `Error: Unsupported expression type: ListComp`
- `Error: For loop must use pl.range(), pl.parallel(), pl.unroll(), pl.pipeline(), pl.while_(), or pl.spmd()`

**Workaround**: All loops inside `@pl.function` bodies (including loops inside `pl.at` scopes) must use one of `pl.range`, `pl.parallel`, `pl.unroll`, `pl.pipeline`, `pl.while_`, `pl.spmd`. List comprehensions are not parsed. Build lists with explicit `pl.range` / `pl.unroll` loops, or replace with loop-carried accumulators.

