# DeepSeek-V4 Single-Layer Decode Flow

One full `Block.forward` pass (model.py:689-701), single card, decode step
(S=1). Tensor shapes use real model dimensions: B=batch, T=B×1, D=7168,
H=128, HEAD_DIM=512, ROPE_DIM=64, Q_LORA=1536, HC=4.

Legend:
- `[orch]`    — orchestrator-only operation (no separate pypto kernel)
- `[EP-orch]` — requires inter-card AllToAllv; host orchestrator calls HCCL

---

## Top-level Block flow

```
═══════════════════════════════════════════════════════════════════════════════
  ENTRY: x  [B, 1, HC=4, D=7168]  bf16
         input_ids  [B, 1]  int64
═══════════════════════════════════════════════════════════════════════════════
                              │
                              ▼
              ╔═══════════════════════════════════════════╗
              ║  attention.py                             ║
              ║  model.py:691-694  (see breakdown below)  ║
              ║                                           ║
              ║  IN : x [B,1,4,D]  bf16                   ║
              ║  OUT: x [B,1,4,D]  bf16                   ║
              ╚═══════════════════════════════════════════╝
                              │
                              ▼
              ╔═══════════════════════════════════════════╗
              ║  moe_router.py                            ║
              ║  model.py:697-698, 564-584                ║
              ║  (hc_pre ffn + ffn_norm fused + gate)     ║
              ║                                           ║
              ║  IN : x [B,1,4,D]  bf16                   ║
              ║  OUT: indices [T, TOPK=6]  int32          ║
              ║       weights [T, TOPK=6]  fp32           ║
              ╚═══════════════════════════════════════════╝
                              │
                              ▼
              ┌───────────────────────────────────────────┐
              │  [EP-orch]  dispatch                      │
              │  pack tokens by dest expert rank          │
              │  AllToAllv → recv_x, recv_expert_id,      │
              │              recv_weights                 │
              └───────────────────────────────────────────┘
                              │
                              ▼
              ╔═══════════════════════════════════════════╗
              ║  moe_expert.py                            ║
              ║  model.py:636-644                         ║
              ║  (local routed + shared)                  ║
              ║                                           ║
              ║  IN : recv_x  [RECV_TOTAL, D]             ║
              ║       recv_expert_id, recv_weights        ║
              ║       x_local [T, D]  (for shared only)   ║
              ║       expert_w1/w2/w3 [N_LOCAL_EXPERTS,…] ║
              ║       shared_w1/w2/w3                     ║
              ║  OUT: recv_y [RECV_TOTAL, D]  bf16        ║
              ║       sh     [T, D]           bf16        ║
              ╚═══════════════════════════════════════════╝
                              │
                              ▼
              ┌───────────────────────────────────────────┐
              │  [EP-orch]  combine                       │
              │  AllToAllv → scatter_add recv_y per token │
              │  → routed_y [T, D]  bf16                  │
              └───────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────────────────┐
              │  [orch]  ffn_out = routed_y + sh          │
              │  model.py:644-645                         │
              └─────────────────────┬─────────────────────┘
                                    │
                                    ▼
              ╔═══════════════════════════════════════════╗
              ║  hc_post.py  (ffn)                        ║
              ║  model.py:700                             ║
              ║                                           ║
              ║  IN : ffn_out [B,1,D], residual [B,1,4,D] ║
              ║       post_ffn [B,1,4], comb_ffn [B,1,4,4]║
              ║  OUT: x_next [B, 1, HC=4, D]  bf16        ║
              ╚═══════════════════════════════════════════╝
                              │
═══════════════════════════════════════════════════════════════════════════════
  EXIT: x_next [B, 1, HC=4, D=7168]  bf16
        → next Block (×61) → MTPBlock → ParallelHead → logits
═══════════════════════════════════════════════════════════════════════════════
```

---

## ATTENTION breakdown

Corresponds to `Block.hc_pre` + `self.attn_norm` + `Attention.forward` +
`Block.hc_post`, model.py:691-694.

```
  IN: x [B, 1, HC=4, D]  bf16
              │
              ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  hc_pre.py  (attn)                                                          ║
║  model.py:691                                                               ║
║                                                                             ║
║  IN :  x          [B, 1, HC=4, D]    bf16                                   ║
║        hc_attn_fn [24, HC*D]         fp32                                   ║
║        hc_attn_scale [3]             fp32                                   ║
║        hc_attn_base  [24]            fp32                                   ║
║  OUT:  x_mixed   [B, 1, D]           bf16  ← 4 copies merged into 1         ║
║        post_attn [B, 1, 4]           fp32  ← saved for hc_post              ║
║        comb_attn [B, 1, 4, 4]        fp32  ← saved for hc_post              ║
╚═════════════════════════════════════════════════════════════════════════════╝
              │ x_mixed [B,1,D]
              │
              ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  mla.py  (attn_norm fused + MLA prolog)                                     ║
║  model.py:692, 496-504                                                      ║
║                                                                             ║
║  IN :  x [B, S, D]                    bf16  (hc_pre output)                 ║
║        norm_w [D]                     fp32  (attn_norm gamma, fused)        ║
║        wq_a [D, Q_LORA=1536]          bf16                                  ║
║        wq_b [Q_LORA, H*HEAD_DIM]      bf16                                  ║
║        wkv  [D, HEAD_DIM=512]         bf16                                  ║
║        rope_cos/sin [T, ROPE_DIM=64]  bf16                                  ║
║        gamma_cq [Q_LORA]              bf16                                  ║
║        gamma_ckv [HEAD_DIM]           bf16                                  ║
║  OUT:  q   [T, H=128, HEAD_DIM=512]   bf16  (RoPE applied)                  ║
║        kv  [T, HEAD_DIM=512]          bf16  (RoPE applied)                  ║
║        qr  [T, Q_LORA=1536]           bf16  (reused by indexer)             ║
╚═════════════════════════════════════════════════════════════════════════════╝
         │ q               │ kv                   │ qr
         │                 │                      │
         │     kv → write ori_kv cache  [orch]    │
         │     ori_kv[block, slot % WIN] = kv     │
         │     model.py:530                       │
         │                 │                      │
         │             ori_kv (PA)                │
         │                 │                      │
         │      ┌──────────┘  (ratio==4 only)     │
         │      │                                 │
         │      │          ╔══════════════════════════════════════════════╗
         │      │          ║  indexer.py                                  ║
         │      │          ║  model.py:402-433                            ║
         │      │          ║                                              ║
         │      │          ║  IN : x [B,1,D], qr [T,Q_LORA]               ║
         │      │          ║       idx_wq_b, weights_proj, cos/sin        ║
         │      │          ║       hadamard_q                             ║
         │      │          ║       inner compressor weights + state(InOut)║
         │      │          ║       idx_kv_cache (InOut,PA), start_pos…    ║
         │      │          ║  OUT: topk_idxs [T, IDX_TOPK=1024]  int32    ║
         │      │          ╚══════════════════════════════════════════════╝
         │      │                      │ topk_idxs
         │      │                      │
         │  (ratio ∈ {4,128})          │
         │      ▼                      │
         │  ╔════════════════════════════════════════════════════════════╗
         │  ║  compressor.py  (main)                                     ║
         │  ║  model.py:316-377                                          ║
         │  ║                                                            ║
         │  ║  IN : x [B,1,D], kv_state/score_state (InOut)              ║
         │  ║       wkv, wgate, ape, weight, cos/sin                     ║
         │  ║       start_pos [B], should_compress [1]                   ║
         │  ║  OUT: cmp_kv_slot [B, HEAD_DIM]  bf16                      ║
         │  ║       ★ orch writes cmp_kv_slot → cmp_kv PA pool           ║
         │  ╚════════════════════════════════════════════════════════════╝
         │              │ cmp_kv (PA pool, updated by orch)
         │              │
         │    [orch]  concat window_topk + topk_idxs
         │    → cmp_sparse_indices [T, WIN+IDX_TOPK=1152]  model.py:507-515
         │              │
         ├─(ratio>0)────┴────────────────────┐
         │                                   │
         ▼ q                                 ▼ q
╔══════════════════════════════════╗  ╔═══════════════════════════════════════╗
║  cfa.py  (ratio ∈ {4, 128})      ║  ║  win_attn.py  (ratio == 0)            ║
║  model.py:528-534                ║  ║  model.py:528-534                     ║
║                                  ║  ║                                       ║
║  IN : q [T,H,HEAD_DIM]           ║  ║  IN : q [T,H,HEAD_DIM]                ║
║       ori_kv/cmp_kv (PA)         ║  ║       ori_kv (PA)                     ║
║       cmp_sparse_indices         ║  ║       window_topk_idxs                ║
║       attn_sink [H]  fp32        ║  ║       attn_sink [H]  fp32             ║
║       seqused_kv [B]             ║  ║       seqused_kv [B]                  ║
║       freqs_cos/sin              ║  ║       freqs_cos/sin                   ║
║  OUT: o [T, H, HEAD_DIM]  bf16   ║  ║  OUT: o [T, H, HEAD_DIM]  bf16        ║
║       (inverse RoPE fused)       ║  ║       (inverse RoPE fused)            ║
╚══════════════════════════════════╝  ╚═══════════════════════════════════════╝
         │ o (both branches merge)
         │
         ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  o_proj.py  (grouped output LoRA)                                           ║
║  model.py:537-542                                                           ║
║                                                                             ║
║  IN :  o    [T, H=128, HEAD_DIM=512]              bf16   (verbatim)         ║
║        wo_a [O_GROUPS, O_LORA=1024, O_GROUP_IN]    bf16   (verbatim;        ║
║             internally reshaped to [G*O_LORA, O_GROUP_IN] as a              ║
║             workaround for hw-native-sys/pypto#1212)                        ║
║        wo_b [D=7168, O_GROUPS*O_LORA=16384]       bf16                      ║
║  OUT:  attn_out  [T, D=7168]  bf16   (view as [B, 1, D] for hc_post)         ║
╚═════════════════════════════════════════════════════════════════════════════╝
              │ attn_out [T, D] → view [B, 1, D]
              │
              ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  hc_post.py  (attn)                                                         ║
║  model.py:694                                                               ║
║                                                                             ║
║  IN :  x        [B, 1, D]          bf16  (attn_out)                         ║
║        residual [B, 1, HC=4, D]    bf16                                     ║
║        post     [B, 1, 4]          fp32                                     ║
║        comb     [B, 1, 4, 4]       fp32                                     ║
║  OUT:  y  [B, 1, HC=4, D]          bf16                                     ║
╚═════════════════════════════════════════════════════════════════════════════╝
              │
  OUT: x [B, 1, HC=4, D]  bf16  → top-level moe_router.py
```

---

## Draft file coverage

| Step | model.py | Draft file | Status |
|---|---|---|---|
| hc_pre (attn) | 691 | `hc_pre.py` | skeleton |
| mla (attn_norm fused + MLA prolog) | 692, 496-504 | `mla.py` | skeleton |
| kv → ori_kv write | 530 | [orch] scatter | — |
| indexer | 402-433 | `indexer.py` | skeleton |
| compressor (main) | 316-377 | `compressor.py` | skeleton |
| window+idx concat | 507-515 | [orch] | — |
| cfa | 528-534 | `cfa.py` | skeleton |
| win_attn | 528-534 | `win_attn.py` | skeleton |
| o_proj | 537-542 | `o_proj.py` | skeleton |
| hc_post (attn) | 694 | `hc_post.py` | skeleton |
| moe_router (hc_pre ffn + ffn_norm + gate) | 697-698, 564-584 | `moe_router.py` | skeleton |
| EP dispatch | — | [EP-orch] HCCL AllToAllv | — |
| moe_expert | 636-644 | `moe_expert.py` | skeleton |
| EP combine | — | [EP-orch] HCCL AllToAllv | — |
| routed_y + sh | 644-645 | [orch] elementwise add | — |
| hc_post (ffn) | 700 | `hc_post.py` | skeleton |

## Layer-type conditional (compress_ratio)

| compress_ratio | Compressor | Indexer | Attention kernel |
|---|---|---|---|
| 0 | not called | not called | win_attn.py |
| 4 | called (ratio=4) | called | cfa.py |
| 128 | called (ratio=128) | not called | cfa.py |

## EP topology notes

- **moe_router**: runs on every card with replicated `gate_w`; indices cover
  global expert space `[0, N_EXPERTS=384)`.
- **moe_expert**: each card holds `N_LOCAL_EXPERTS = N_EXPERTS / EP_WORLD_SIZE`
  routed expert weights. `recv_x` is the post-dispatch token set (source:
  all cards); `x_local` is the original pre-dispatch local token set (shared
  expert only). The two inputs are distinct token populations.
- **shared expert**: computed locally on `x_local` with no communication;
  result `sh` stays on the card and is added after combine.
- **hc_post** (both attn and ffn): same draft file, called twice per Block
  with different post/comb tensors from the respective hc_pre call.
