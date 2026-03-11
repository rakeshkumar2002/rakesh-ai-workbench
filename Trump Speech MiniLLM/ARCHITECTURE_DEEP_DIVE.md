# Architecture Deep Dive (MiniLLM)

This document explains the internal architecture implemented in `Class 5.ipynb`.

## High-Level Computation

Given token IDs `idx` with shape `[B, T]`:

1. Token embedding: `x = Embedding(idx)` -> `[B, T, d_model]`
2. Pass through `n_layers` transformer blocks
3. Final RMSNorm
4. LM head projection to logits `[B, T, vocab_size]`
5. If targets are provided, compute cross-entropy loss over flattened logits/targets

No encoder is used. This is a causal decoder-only language model.

## Block Structure (Pre-Norm Transformer)

Each block applies:

1. `x = x + Attention(RMSNorm(x))`
2. `x = x + FFN(RMSNorm(x))`

This is pre-norm residual design, which tends to be more stable in deeper transformers.

## RMSNorm

RMSNorm normalizes by root-mean-square without mean subtraction:

`RMS(x) = sqrt(mean(x^2) + eps)`

Output:

`y = (x / RMS(x)) * w`

- One learnable scale vector `w` of size `d_model`
- No bias term
- Used before attention and FFN, plus one final norm before LM head

## RoPE (Rotary Position Embeddings)

RoPE rotates Q/K channel pairs according to token position and frequency:

- Frequencies are precomputed up to `max_seq_len`
- Applied to Q and K (not V)
- Encodes relative positional information via phase differences in dot products

Implementation details:

- `precompute_rope_freqs(head_dim, max_seq_len)` returns cosine and sine tables
- `apply_rope(q_or_k, cos, sin)` rotates even/odd channel pairs

## Grouped Query Attention (GQA)

Configuration:

- Query heads: `n_heads = 8`
- KV heads: `n_kv_heads = 2`
- Replication factor: `n_rep = n_heads // n_kv_heads = 4`

Flow:

1. Project input into Q, K, V
2. Reshape:
   - `Q: [B, n_heads, T, head_dim]`
   - `K,V: [B, n_kv_heads, T, head_dim]`
3. Apply RoPE to Q and K
4. Repeat K/V heads by `n_rep` to align with query-head count
5. Compute scaled dot-product attention with causal mask
6. Project back to `d_model`

Why GQA:

- Smaller KV representation than full MHA
- Better memory/runtime profile, especially for long context and generation

## SwiGLU Feed-Forward

The FFN uses gated activation:

- `gate = SiLU(xW_gate)`
- `up = xW_up`
- `hidden = gate * up`
- `out = hiddenW_down`

Plus dropout (`DROPOUT = 0.2`) in FFN output path.

## Parameter Breakdown (Approx, vocab = 8192)

Using:

- `d_model = 512`
- `n_layers = 10`
- `n_heads = 8`, `n_kv_heads = 2`, `head_dim = 64`
- `ffn_hidden_dim = 1536`

Approx trainable counts:

- Embedding/LM head (tied): `8192 * 512 = 4,194,304`
- Per transformer block:
  - Attention projections: `655,360`
  - FFN projections: `2,359,296`
  - Norm weights: `1,024`
  - Total per block: `3,015,680`
- 10 blocks: `30,156,800`
- Final norm: `512`

Total trainable params:

- `~34,351,616` (`~34.35M`)

Note: RoPE cosine/sine tables are buffers, not trainable parameters.

## Training and Context Behavior

Current training profile uses:

- `CONTEXT_LEN = 256` for faster iteration
- Model `max_seq_len = 512` still defines RoPE table size and generation cap

This means training sees 256-token windows, while generation can still use up to 512-token rolling context.

## Generation Behavior

`generate(model, prompt, max_new_tokens, temperature)`:

1. Encode prompt with BPE
2. Iterate token-by-token
3. Use latest context slice `tokens[:, -max_seq_len:]`
4. Sample next token from temperature-scaled softmax
5. Decode to text at the end

This is standard autoregressive next-token sampling.
