# mini-llm-from-scratch

## Overview
This repository is a **step-by-step, educational implementation of a GPT-style Large Language Model (LLM) built entirely from scratch using PyTorch**. The project is designed to teach *how LLMs actually work internally*, not just how to use existing libraries.

The model you build here follows the same **core architecture** as GPT-2 / GPT-3 (decoder-only Transformer with causal self-attention), but at a **small, understandable scale** suitable for learning, experimentation, and research prototyping.

---

## Objectives

- Understand the **end-to-end LLM pipeline**: text → tokens → embeddings → attention → logits → text
- Implement **every major component manually** (no HuggingFace Trainer, no prebuilt models)
- Gain intuition about **why each design choice exists**
- Create a clean base for extending into:
  - BPE tokenization
  - Multi-head attention
  - RoPE / RMSNorm
  - Instruction fine-tuning
  - Alignment (RLHF-style)

---

## What This Repository Builds

By the end, you will have:

- A working **GPT-style language model**
- A complete **training loop for next-token prediction**
- An **autoregressive text generation pipeline**
- Modular code that mirrors real LLM implementations

This is a **learning LLM**, not a production-scale model.

---

## Repository Structure

```
mini-llm-from-scratch/
│
├── data/
│   ├── input.txt            # Raw training text
│   └── prepare_data.py      # Data loading & preprocessing
│
├── tokenizer/
│   ├── char_tokenizer.py    # Character-level tokenizer
│   └── __init__.py
│
├── model/
│   ├── embeddings.py        # Token & positional embeddings
│   ├── attention.py         # Causal self-attention
│   ├── transformer.py       # Transformer block
│   ├── gpt.py               # Full GPT-style model
│   └── __init__.py
│
├── training/
│   ├── dataset.py           # Batch sampling logic
│   ├── train.py             # Training loop
│   └── eval.py              # Loss evaluation utilities
│
├── generation/
│   ├── generate.py          # Autoregressive text generation
│   └── sampling.py          # Temperature / top-k sampling
│
├── configs/
│   └── base_config.py       # Model & training hyperparameters
│
├── notebooks/
│   └── walkthrough.ipynb    # Step-by-step explanation notebook
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Step-by-Step Learning Path

### Step 1: Tokenization
- Convert raw text into integer token IDs
- Learn why LLMs operate purely on numbers

### Step 2: Dataset Construction
- Build (input, target) pairs for next-token prediction
- Understand causal language modeling

### Step 3: Embeddings
- Token embeddings
- Positional embeddings
- Why order information matters

### Step 4: Self-Attention
- Query, Key, Value projections
- Causal masking
- Why attention scales better than RNNs

### Step 5: Transformer Block
- Residual connections
- Layer normalization
- Feed-forward expansion

### Step 6: GPT Model
- Stack transformer blocks
- Language modeling head
- Loss computation

### Step 7: Training
- Forward pass
- Backpropagation
- Optimization with AdamW

### Step 8: Generation
- Autoregressive decoding
- Sampling strategies

---

## How This Relates to Real LLMs

| Concept | This Repo | Production LLMs |
|------|---------|----------------|
| Architecture | GPT-style decoder | Same |
| Attention | Single-head | Multi-head + FlashAttention |
| Tokenizer | Character-level | BPE / SentencePiece |
| Scale | Thousands of params | Billions / Trillions |
| Alignment | Not included | SFT + RLHF |

---

## Requirements

```
torch>=2.0
```

Python 3.9+ recommended.

---

## How to Run

1. Prepare data:
```bash
python data/prepare_data.py
```

2. Train the model:
```bash
python training/train.py
```

3. Generate text:
```bash
python generation/generate.py
```

---

## Who This Repo Is For

- Students learning **Transformers and LLM internals**
- Researchers who want a **clean experimental base**
- Engineers transitioning from *using* LLMs to *building* them

---

## Extensions (Recommended)

- Replace char tokenizer with **BPE**
- Add **multi-head attention**
- Implement **RoPE**
- Add **KV cache for fast inference**
- Instruction fine-tuning

---

## License

MIT License

---

## Disclaimer

This project is for **educational and research purposes only**. It is not intended for deployment in production systems.

