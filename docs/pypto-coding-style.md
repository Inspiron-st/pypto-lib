# PyPTO Coding Style

This document describes the canonical coding style for writing `pl.function`
kernels in PyPTO-Lib.

```python
import pypto.language as pl
```

`pl` is the only accepted module alias.

---

## 1. Program Structure and `pl.at` Scopes

PyPTO-Lib kernels are written as **opaque** functions. The frontend does not
draw the InCore / Orchestration boundary explicitly; instead, each compute
region is wrapped in `with pl.at(level=pl.Level.CORE_GROUP, ...)` and the
compiler lowers that region to InCore. Code outside any `pl.at` block stays in
orchestration (host/AICPU control flow).

```python
@pl.program
class Qwen3Decode:
    @pl.function(type=pl.FunctionType.Opaque)
    def qwen3_decode(self, hidden_states: pl.Tensor[..., pl.BF16], ...):
        # orchestration code (loops, tensor allocation)
        for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
                # InCore region — vector / cube / mte ops
                ...
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
                ...
        return out
```

### `pl.at` parameters

| Parameter | Required | Purpose |
|-----------|----------|---------|
| `level=pl.Level.CORE_GROUP` | yes | Lowering target. `CORE_GROUP` is the only level used in pypto-lib. |
| `name_hint="..."` | recommended | Stable label for the region. Appears in generated kernel filenames and profiling traces; aids per-region debugging. |

`pl.at` blocks may nest: an outer `pl.at` defining the InCore scope, with
inner `pl.at` blocks (each with its own `name_hint`) splitting it into named
sub-kernels.

---

## 2. Vector Ops

Run on the vector unit, inside an InCore region (a `pl.at` block, a
`pl.spmd` body, or a `pl.parallel(chunk=N)` body — see §5). Vector ops are
the standard tools for the cast / activation / norm epilogue around a
matmul, and for small standalone reductions.

### Elementwise

Tensor-tensor binary: `pl.add`, `pl.sub`, `pl.mul`, `pl.div`,
`pl.maximum`, `pl.minimum`. Tensor-scalar variants (second operand is a
Python `int`/`float` or `pl.Scalar`) suffix with `s`: `pl.adds`, `pl.subs`,
`pl.muls`, `pl.divs`, `pl.maxs`, `pl.mins`. Unary: `pl.neg`, `pl.abs`,
`pl.exp`, `pl.log`, `pl.sqrt`, `pl.recip`, `pl.rsqrt`. Activations:
`pl.relu`, `pl.lrelu`, `pl.prelu`. Type conversion: `pl.cast(x, target_type=...)`.
Binary ops broadcast over compatible shapes; prefer `pl.recip` + `pl.mul`
over `pl.div` on hot paths.

```python
silu_x = pl.mul(x, pl.recip(pl.add(pl.exp(pl.neg(x)), one)))   # x * sigmoid(x)
out_bf16 = pl.cast(acc_fp32, target_type=pl.BF16)
scaled = pl.muls(scores, attn_scale)                           # scalar mul
```

For comparison / select / bit-twiddling — `pl.cmp`, `pl.cmps`, `pl.sel`,
`pl.sels`, `pl.and_`, `pl.or_`, `pl.xor`, `pl.not_`, `pl.shl`, `pl.shr` —
see existing kernels.

### Reductions

Row reductions (along the last axis, return `[..., 1]`): `pl.row_max`,
`pl.row_min`, `pl.row_sum`. Column reductions (return `[1, ...]`):
`pl.col_max`, `pl.col_min`, `pl.col_sum`. Pair with broadcast ops below for
the typical RMSNorm / softmax patterns:

```python
sq_sum = pl.row_sum(pl.mul(x, x))                       # [B, 1]
inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))
```

### Row / column broadcast

Apply a column to each row: `pl.row_expand_add`, `pl.row_expand_sub`,
`pl.row_expand_mul`, `pl.row_expand_div`. Apply a row to each column:
`pl.col_expand_sub`, `pl.col_expand_mul`, `pl.col_expand_div`. Each maps
to a single hardware broadcast op — use them rather than reshaping a
vector and relying on elementwise broadcast.

```python
# RMSNorm body: normed[i, j] = x[i, j] * inv_rms[i] * gamma[j]
normed = pl.col_expand_mul(pl.row_expand_mul(x, inv_rms), gamma)
```

### Fill and pad

`pl.full(shape, dtype=..., value=...)` allocates a scalar-filled
tensor/tile (typical use: zero-init a partial accumulator before a
reduction). `pl.fillpad(x, pad_value=...)` rewrites the padded tail of a
`valid_shape` slice with a sentinel — most often `pl.PadValue.min` to
mask out invalid positions before a softmax `row_max`. There is also an
in-place `pl.fillpad_inplace`.

```python
partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
scores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)   # -inf in tail
```

---

## 3. Cube Ops

Matrix multiply primitives. They run on the cube unit, inside an InCore
region (a `pl.at` block, a `pl.spmd` body, or a `pl.parallel(chunk=N)`
body — see §5), and produce FP32 results by default (`out_dtype` may
override).

### `pl.matmul(lhs, rhs, *, out_dtype=None, a_trans=False, b_trans=False)`

Plain matmul. `a_trans` / `b_trans` transpose the corresponding operand
without a separate `pl.transpose`. `out_dtype` overrides the default FP32
accumulator dtype when set.

```python
out = pl.matmul(tile_a, tile_b)
out = pl.matmul(tile_a, tile_b, out_dtype=pl.FP32, b_trans=True)
```

### `pl.matmul_acc(acc, lhs, rhs, *, a_trans=False, b_trans=False)`

Fused multiply-accumulate: `acc += lhs @ rhs`. Use this inside a K-loop to
keep the partial sum on chip. The first iteration uses `pl.matmul`,
subsequent iterations use `pl.matmul_acc`. The `acc` destination is
allocated outside `pl.at` (see §4):

```python
acc = pl.create_tensor([M, N], dtype=pl.FP32)
with pl.at(level=pl.Level.CORE_GROUP, name_hint="kproj"):
    for kb in pl.pipeline(0, K_BLOCKS, stage=2):
        k0 = kb * K_STEP
        tile_a = pl.slice(a, [M, K_STEP], [m0, k0])
        tile_b = pl.slice(b, [K_STEP, N], [k0, n0])
        if kb == 0:
            acc = pl.matmul(tile_a, tile_b)
        else:
            acc = pl.matmul_acc(acc, tile_a, tile_b)
```

### `pl.matmul_bias(lhs, rhs, bias)`

Matmul fused with a bias add: `lhs @ rhs + bias`. Cheaper than a separate
`pl.add` epilogue when the bias is broadcast over the M axis.

### `pl.batch_matmul`, `pl.batch_matmul_acc`

Batched variants for stacked matmul (shape `[B, M, K] @ [B, K, N]`); same
arg shape as the non-batched forms.

### `pl.gemv`, `pl.gemv_acc`, `pl.gemv_bias`

Vector-matrix specializations (1-row left operand). Prefer over `pl.matmul`
when M is 1 — the cube schedules the smaller form more efficiently.

---

## 4. MTE Ops (Data Movement / Shape)

MTE primitives manipulate tensor views and stage data without explicit
load/store. The compiler decides where the actual TLOAD/TSTORE land based
on where each `pl.slice` / `pl.assemble` sits relative to `pl.at`.

### `pl.create_tensor(shape, dtype=...)` — orchestration only

`create_tensor` lives **outside** `pl.at`. Use it when multiple `pl.at`
regions cooperate to fill one intermediate tensor — the tensor is allocated
once in orchestration, then each region writes its piece via `assemble`. If
the result of a single `pl.at` flows directly to its caller without further
assembly, no `create_tensor` is needed.

```python
# Multi-stage assembly: q_proj is built by per-tile assembles
q_proj = pl.create_tensor([batch_padded, q_hidden], dtype=pl.BF16)
for q0 in pl.parallel(0, q_hidden, Q_OUT_STEP):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
        ...
        q_proj = pl.assemble(q_proj, q_acc, [b0, q0])
```

### `pl.slice` / `pl.assemble` — load/store at the `pl.at` boundary

`pl.slice(tensor, sizes, offsets, valid_shape=...)` takes a sub-region;
`pl.assemble(dst, src, offsets)` writes a sub-region back. When these cross
the `pl.at` boundary they lower to InCore TLOAD / TSTORE — the tensor
descriptor (offset / shape / stride) is passed as an InCore argument and
the data movement is generated by the compiler. Argument order on `slice`
is **(sizes, offsets)**.

```python
tile = pl.slice(hidden_states, [BATCH_TILE, K_STEP], [b0, k0])
...
q_proj = pl.assemble(q_proj, q_acc, [b0, q0])
```

The shorthand subscript forms are equivalent and often clearer:

```python
tile = hidden_states[b0 : b0 + BATCH_TILE, k0 : k0 + K_STEP]   # slice
q_proj[b0 : b0 + BATCH_TILE, q0 : q0 + Q_OUT_STEP] = q_acc     # assemble
```

`valid_shape=[real_h, real_w]` on a slice marks a padded load: the slice
has nominal size `sizes` but only the leading `valid_shape` rows/cols
carry real data, and the compiler zero-pads the tail. Use this for
dynamic batch / sequence length when the kernel works on a fixed
`BATCH_TILE` but the caller may pass fewer valid rows.

### `pl.reshape(x, new_shape)`

Logical reshape (view-only). Total element count must match. Legal both
inside and outside `pl.at`.

```python
flat = pl.reshape(q_chunk, [BATCH_TILE * H, D])
```

---

## 5. Loops: `pl.range`, `pl.parallel`, `pl.pipeline`, `pl.spmd`

PyPTO has four loop constructs. The choice between them is **semantic** —
it tells the compiler what scheduling and codegen are valid.

Each construct has a fixed placement relative to `pl.at`:

| Construct | Outside `pl.at` (orchestration) | Inside `pl.at` (InCore) |
|-----------|:-:|:-:|
| `pl.range` | yes | yes |
| `pl.parallel` | yes | no |
| `pl.pipeline` | no | yes |
| `pl.spmd` | yes (body is implicitly InCore) | no |

`pl.parallel` distributes iterations across cores, so it must sit in
orchestration. `pl.pipeline` software-pipelines stages within a single
InCore region, so it must sit inside `pl.at`. `pl.range` is a plain
sequential loop and is legal in either place. `pl.spmd` is a parallel SPMD
loop that bundles its own InCore region — see below.

`pl.range`, `pl.parallel`, and `pl.pipeline` share the same positional-arg
shape, mirroring Python's `range`:

```text
pl.<loop>(stop)
pl.<loop>(start, stop)
pl.<loop>(start, stop, step)
```

Each argument may be either a Python `int` or a `pl.Scalar`.

### `pl.range` — sequential

Iterations execute in strict order. Loop-carried dependencies are allowed.

```python
for kb in pl.range(HIDDEN_BLOCKS):              # range [0, HIDDEN_BLOCKS)
for kb in pl.range(1, hidden_blocks):           # range [1, hidden_blocks)
for k0 in pl.range(0, HIDDEN, K_STEP):          # start, stop, step
```

To carry state across iterations, pass `init_values=` and unpack the loop
variable as a `(idx, (state...))` tuple:

```python
for i, (acc,) in pl.range(N, init_values=(zero,)):
    acc = pl.add(acc, x[i])
```

### `pl.parallel` — independent iterations

Iterations are guaranteed independent — the compiler may split, reorder, or
schedule them across cores. Same arg shape and `init_values` support as
`pl.range`.

```python
for b in pl.parallel(BATCH):                          # short form: extent only
for b0 in pl.parallel(0, batch_padded, BATCH_TILE):   # start, stop, step
```

### `pl.pipeline` — software-pipelined sequential

Sequential like `pl.range`, but the compiler software-pipelines successive
iterations across compute and memory units. `stage=N` (required keyword) is
the pipeline depth — the loop body is replicated `stage` times for
ping-pong buffering; the outer trip count advances in strides of
`stage * step` and a tail dispatch covers the remainder when the trip count
is not divisible by `stage`. Typical values are 2 or 4.

```python
for kb in pl.pipeline(HIDDEN_BLOCKS, stage=2):
for kb in pl.pipeline(2, HIDDEN_BLOCKS, stage=2):     # start at 2
for kb in pl.pipeline(0, input_proj_k_blocks, stage=4):
```

`init_values=` is supported, with the same `(idx, (state...))` unpacking as
`pl.range`. Use `pl.pipeline` for the inner reduction loop of a matmul (the
K loop) — each iteration loads a new tile of the left/right operand and
accumulates into the same output.

### `pl.spmd` — parallel SPMD dispatch

`pl.spmd(core_num)` dispatches `core_num` blocks in parallel; iteration
starts at 0 and steps by 1 (only the block count is positional, **not**
start/stop/step). Two forms:

**Loop form** — body is auto-outlined into a synthetic InCore function;
the iteration variable binds the per-block index (equivalent to
`pl.tile.get_block_idx()`). No surrounding `pl.at` is needed or allowed:

```python
for ob0 in pl.spmd(Q_SPMD_BLOCKS, name_hint="q_proj"):
    # implicit InCore region — vector / cube / mte ops here
    for ob in pl.range(ob0 * 4, (ob0 + 1) * 4):
        q0 = ob * Q_OUT_STEP
        ...
```

**Context-manager form** — body must be a single call to a pre-defined
InCore kernel:

```python
with pl.spmd(4):
    out = self.kernel(a, b, out)
```

Keyword args:

| Kwarg | Default | Purpose |
|-------|---------|---------|
| `sync_start` | `False` | If True, all blocks start execution simultaneously. |
| `name_hint` | `""` | Stable label for the outlined function. |

### `pl.parallel(chunk=N)` — sugar for parallel + InCore + in-chunk loop

`pl.parallel(..., chunk=N)` partitions the iteration range into groups of
`N` and is shorthand for an outer parallel chunk loop, an InCore region per
chunk, and an inner sequential in-chunk loop:

```python
for b in pl.parallel(0, BATCH, 1, chunk=4):
    # body runs InCore for each iteration
    ...
```

is equivalent to:

```python
for b_chunk in pl.parallel(0, BATCH, 4):
    with pl.at(level=pl.Level.CORE_GROUP):
        for b in pl.range(b_chunk, b_chunk + 4):
            # same body
            ...
```

Use the sugared form when each chunk is a self-contained InCore region with
no further sub-stages; expand to the explicit form when you need named
sub-regions (each with its own `name_hint`) inside the chunk.

---

## 6. Mixed Cube + Vector Within One `pl.at`

A single `pl.at` region can contain both cube (matmul) and vector (cast,
add, row_sum, …) ops. The compiler assigns each op to the appropriate unit
and pipelines cube/vec where possible. This is the standard pattern for a
projection with cast/residual epilogue:

```python
q_proj = pl.create_tensor([batch_padded, hidden], dtype=pl.BF16)
for q0 in pl.parallel(0, hidden, Q_OUT_STEP):
    q_acc = pl.create_tensor([BATCH_TILE, Q_OUT_STEP], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
        for kb in pl.pipeline(0, input_proj_k_blocks, stage=2):
            k0 = kb * INPUT_PROJ_K_STEP
            tile_a = pl.slice(normed_tile,
                              [BATCH_TILE, INPUT_PROJ_K_STEP], [0, k0])  # vec source
            tile_b = pl.slice(wq,
                              [INPUT_PROJ_K_STEP, Q_OUT_STEP], [k0, q0])  # cube right
            if kb == 0:
                q_acc = pl.matmul(tile_a, tile_b)              # cube
            else:
                q_acc = pl.matmul_acc(q_acc, tile_a, tile_b)   # cube
        q_bf16 = pl.cast(q_acc, target_type=pl.BF16)           # vector
        q_proj = pl.assemble(q_proj, q_bf16, [b0, q0])         # mte
```

Larger fused regions (RMSNorm + projection + residual) follow the same
shape: a `pl.pipeline` matmul reduction, then the vector epilogue, then
`assemble` back to GM.
