# Trump Speech MiniLLM: End-to-End Overview

This project trains a decoder-only transformer language model on Trump rally speech text using Byte Pair Encoding (BPE) tokenization.

## 1) Data Pipeline

- Source: Kaggle dataset `christianlillelund/donald-trumps-rallies`
- All `.txt` files are loaded and concatenated into one training corpus.
- The corpus is split into:
  - `train_data`: first 90%
  - `val_data`: last 10%

## 2) Tokenization (BPE)

- Tokenizer type: Hugging Face `tokenizers` Byte-Level BPE
- Target vocab size: `8192`
- Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`
- Persistence:
  - Path: `./tokenizer_artifacts/trump_bpe_tokenizer.json`
  - If file exists: load
  - Else: train and save

Core interfaces:

- `encode(text: str) -> list[int]`
- `decode(ids: list[int]) -> str`

This replaces earlier character-level tokenization and usually reduces sequence length per sample.

## 3) Model Summary

Architecture: decoder-only transformer with:

- RMSNorm (pre-norm)
- RoPE (rotary position embeddings)
- GQA (Grouped Query Attention)
- SwiGLU feed-forward network
- Causal masking (next-token prediction)
- Tied token embedding and LM head weights

Current config:

- `d_model = 512`
- `n_layers = 10`
- `n_heads = 8`
- `n_kv_heads = 2`
- `head_dim = 64`
- `ffn_hidden_dim = 1536`
- `max_seq_len = 512`
- `vocab_size = tokenizer.get_vocab_size()` (typically near 8192)

Approximate trainable parameters (for vocab = 8192):

- About `34.35M` trainable params

## 4) Training Setup (Current Fast Profile)

- `BATCH_SIZE = 8`
- `GRAD_ACCUM_STEPS = 2` (effective batch = 16 sequences/optimizer step)
- `CONTEXT_LEN = 256` (training chunk length)
- `LEARNING_RATE = 2e-4`
- `MAX_STEPS = 2000`
- `EVAL_INTERVAL = 200`
- `EVAL_STEPS = 10`
- Gradient clipping: `max_norm = 1.0`
- Optimizer: `AdamW`

Objective:

- Predict token `t+1` from tokens up to `t`
- Loss: cross-entropy over vocabulary

## 5) Inference / Text Generation

Generation is autoregressive:

1. Encode prompt into token IDs.
2. Repeatedly:
   - Run model on most recent context (`<= max_seq_len`)
   - Sample next token from softmax(logits / temperature)
   - Append sampled token
3. Decode token IDs back to text.

Temperature guidance:

- Lower: more conservative/repetitive
- Higher: more diverse/chaotic

## 6) Key Design Choices

- BPE instead of character tokens improves efficiency and token-level expressiveness.
- GQA reduces KV memory and attention compute compared with full multi-head KV per query head.
- RoPE provides positional information without learned absolute positional embeddings.
- Pre-norm RMSNorm improves training stability in deeper stacks.
