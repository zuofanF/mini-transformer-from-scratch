# Transformer Architecture Details

## Overview

This is a complete Transformer implementation for English-to-Japanese translation, built from scratch using PyTorch.

## Core Components

### 1. Scaled Dot-Product Attention

Attention(Q, K, V) = softmax(QK^T / √d_k)V

- Dimension scaling by √d_k prevents softmax saturation
- Mask support for padding and causal masking

### 2. Multi-Head Attention

- 8 parallel attention heads
- Each head: 256/8 = 32 dimensions
- Allows attending to different representation subspaces

### 3. Feed-Forward Network

FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

- Inner dimension: 512
- Two linear layers with ReLU activation

### 4. Layer Normalization & Residual Connections

- Pre-normalization for training stability
- Residual connections enable deeper networks

## Model Hyperparameters

- d_model: 256
- num_heads: 8
- num_layers: 3 (both encoder and decoder)
- d_ff: 512
- dropout: 0.1
- max_seq_len: 100

## Training Configuration

- batch_size: 32
- learning_rate: 0.0001
- optimizer: Adam
- epochs: 30
- Training time: ~5 minutes on Tesla T4 GPU

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
