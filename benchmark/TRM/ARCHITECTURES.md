# TRM Architecture Reference

All backbones share the same recursive reasoning loop. They differ in two places:
1. **Encoder** — how raw input becomes a `(B, S, D)` tensor
2. **Reasoning block** — what operation is applied inside the recursive core

## Notation

| Symbol | Meaning |
|--------|---------|
| `B` | Batch size |
| `S` | Sequence length (= `num_features` for tabular, `pool_k` for GNN) |
| `D` | Hidden dimension (`hidden_dim`) |
| `K` | Number of attention-pooling queries (`pool_k`, GNN only) |
| `N` | Total nodes across all graphs in a PyG batch |
| `E` | Total edges across all graphs in a PyG batch |

---

## Recursive Reasoning Loop (shared by all backbones)

```
init:
    pred  = learnable (1, S, D) expanded to (B, S, D)
    Z     = learnable (1, S, D) expanded to (B, S, D)

for h = 1..H_cycles:
    inj = pred + encoded                     # (B, S, D)
    for l = 1..L_cycles:
        Z = ReasoningModule(Z, inj)          # inject then apply L_layers blocks
    pred = ReasoningModule(pred, Z)

output = Linear(LayerNorm(pred.mean(dim=1))) # (B, D) → (B, 1) → squeeze → (B,)
```

Only the final H cycle receives gradient; earlier cycles run under `torch.no_grad()`.

The block is applied `L_layers * (L_cycles + 1) * H_cycles` times total per forward pass (e.g. 2 layers, 2 inner, 3 outer = 18 block applications).

---

## 1. `"transformer"` — Tabular + Transformer Reasoning

### Encoder
```
Input: (B, num_features) float tensor

Per-feature projection:
    for i in 0..S-1:
        feat_i = Linear(1 → D)(x[:, i:i+1])    # (B, 1) → (B, D)
    stack → (B, S, D)
    + pos_embed (1, S, D)                        # learnable positional embedding

Output: (B, S, D)      where S = num_features
```

### Reasoning Block: TransformerBlock
```
forward(x):                                      # (B, S, D)
    h = LayerNorm(x)
    x = x + MultiHeadAttention(h, h, h)          # self-attention, num_heads heads
    x = x + FFN(LayerNorm(x))                    # 2-layer FFN with GELU
    return x                                      # (B, S, D)
```

---

## 2. `"mlp"` — Tabular + MLP Reasoning

### Encoder
Same as `"transformer"` (per-feature projection + positional embedding).

### Reasoning Block: MLPBlock
```
forward(x):                                      # (B, S, D)
    x = x + FFN(LayerNorm(x))                    # 2-layer FFN with GELU
    return x                                      # (B, S, D)
```

No inter-position communication — each position is processed independently.

---

## 3. `"gcn"` — Tabular + GCN Reasoning

### Encoder
Same as `"transformer"` (per-feature projection + positional embedding).

### Reasoning Block: GCNBlock (with learnable sparse adjacency)

**Learnable Adjacency** (shared across all GCN layers):
```
adj_logits: Parameter(S, S)               # learned
A = softmax(adj_logits, dim=-1)           # row-wise softmax → soft adjacency
A = top_k_sparsify(A, k=gcn_adj_k)       # keep top-k per row, zero rest
if training:
    A = DropEdge(A, p=gcn_drop_edge)      # randomly zero entries
A = row_normalize(A)                      # re-normalize rows to sum to 1
```

**Block forward**:
```
forward(x):                                      # (B, S, D)
    h = LayerNorm(x)                             # pre-norm
    h = ReLU(A @ h)                              # GCN aggregation: (S,S) @ (B,S,D) → (B,S,D)
    h = Linear(D → D)(h)                         # learnable weight matrix
    gate = sigmoid(Linear(2D → D)([x; h]))       # gated residual
    x = x + gate * h
    x = x + FFN(LayerNorm(x))                    # standard FFN sub-layer
    return x                                      # (B, S, D)
```

**Anti-oversmoothing strategies** (the block is applied up to ~18+ times):
- **Gated residual**: learned gate can bypass GCN when output is uninformative
- **DropEdge** (p=0.3): random edge removal during training slows convergence to uniform
- **Top-k adjacency** (k=8): limits spectral gap, reduces diffusion rate
- **Skip connections**: block-level `x + ...` residuals already present

### Config
| Parameter | Default | Description |
|-----------|---------|-------------|
| `gcn_adj_k` | 8 | Top-k neighbors per node in learnable adjacency |
| `gcn_drop_edge` | 0.3 | DropEdge probability during training |
| `gcn_gate_residual` | True | Gated residual (True) vs simple additive residual (False) |

---

## 4. `"gnn_transformer"` — GNN Encoder + Transformer Reasoning

### Encoder
```
Input: PyG Batch (z, edge_index, edge_attr, batch)

GNNEncoder (CGCNN-style):
    atom_embed: Embedding(max_atomic_num → D)     # z: (N,) → (N, D)
    edge_feat: GaussianExpansion(edge_attr)        # (E, 1) → (E, n_gaussians)
              → Linear(n_gaussians → D)            # (E, D)
    for layer in CGConv layers (gnn_conv_layers):
        x = CGConv(x, edge_index, edge_feat)       # message passing
        x = BatchNorm(x)
        x = ReLU(x)
    Output: (N, D) node embeddings

AttentionPooling:
    queries: learnable (1, K, D)                   # K = pool_k
    Pad node embeddings → (B, max_nodes, D)
    CrossAttention(queries, padded_nodes)           # (B, K, D)
    LayerNorm → Output: (B, K, D)

Output: (B, S, D)      where S = pool_k (K)
```

### Reasoning Block
Same as `"transformer"` (TransformerBlock with self-attention).

---

## 5. `"gnn_mlp"` — GNN Encoder + MLP Reasoning

### Encoder
Same as `"gnn_transformer"` (GNNEncoder + AttentionPooling).

### Reasoning Block
Same as `"mlp"` (MLPBlock, no inter-position communication).

---

## 6. `"gnn_gcn"` — GNN Encoder + GCN Reasoning

### Encoder
Same as `"gnn_transformer"` (GNNEncoder + AttentionPooling).

Here `S = pool_k`, so the learnable adjacency is `(pool_k, pool_k)` — treating the K pooled graph tokens as nodes in a synthetic graph.

### Reasoning Block
Same as `"gcn"` (GCNBlock with shared learnable sparse adjacency).

---

## Shape Summary

| Backbone | Input | Encoded | S | Reasoning Block |
|----------|-------|---------|---|-----------------|
| `transformer` | `(B, num_features)` | `(B, num_features, D)` | `num_features` | TransformerBlock |
| `mlp` | `(B, num_features)` | `(B, num_features, D)` | `num_features` | MLPBlock |
| `gcn` | `(B, num_features)` | `(B, num_features, D)` | `num_features` | GCNBlock |
| `gnn_transformer` | PyG Batch | `(B, pool_k, D)` | `pool_k` | TransformerBlock |
| `gnn_mlp` | PyG Batch | `(B, pool_k, D)` | `pool_k` | MLPBlock |
| `gnn_gcn` | PyG Batch | `(B, pool_k, D)` | `pool_k` | GCNBlock |

## FFN (shared sub-layer)

Used in all blocks (TransformerBlock, MLPBlock, GCNBlock):
```
Linear(D → D * ffn_expansion) → GELU → Dropout → Linear(D * ffn_expansion → D) → Dropout
```
