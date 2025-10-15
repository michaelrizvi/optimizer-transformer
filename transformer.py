import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embedding to query and key tensors."""
    # q, k shape: (batch_size, n_heads, seq_len, head_dim)
    # cos, sin shape: (1, 1, seq_len, head_dim//2)
    
    # Split the last dimension into pairs
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    
    # Apply rotation: [cos*q1 - sin*q2, sin*q1 + cos*q2]
    q_embed = torch.cat([
        q1 * cos - q2 * sin,
        q1 * sin + q2 * cos
    ], dim=-1)
    
    k_embed = torch.cat([
        k1 * cos - k2 * sin,
        k1 * sin + k2 * cos
    ], dim=-1)
    
    return q_embed, k_embed


def precompute_freqs_cis(dim, max_len, base=10000.0):
    """Precompute the frequency tensor for rotary embeddings."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, cos=None, sin=None):
        B, T, C = x.size()
        qkv = self.qkv(x).reshape(B, T, self.n_heads, 3 * self.d_k).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)

        # Apply RoPE if cos and sin are provided
        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn_weights) @ v

        attn = attn.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out(attn)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, cos=None, sin=None):
        x = x + self.attn(self.ln1(x), cos=cos, sin=sin)
        x = x + self.ff(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout=0.1, 
                 position_encoding_type="absolute", rope_base=10000.0):
        super().__init__()
        self.position_encoding_type = position_encoding_type
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_len = max_len
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # Initialize positional encoding based on type
        if position_encoding_type == "absolute":
            self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        elif position_encoding_type == "rotary":
            head_dim = d_model // n_heads
            cos, sin = precompute_freqs_cis(head_dim, max_len, rope_base)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)
        elif position_encoding_type == "none":
            pass
        else:
            raise ValueError(f"Unknown position_encoding_type: {position_encoding_type}")
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, position_ids=None):
        B, T = x.size()
        
        # Token embeddings
        token_embeddings = self.token_emb(x)
        
        if self.position_encoding_type == "absolute":
            # Standard sequential position embeddings
            position_embeddings = self.pos_emb[:, :T, :]
            x = token_embeddings + position_embeddings
            cos, sin = None, None
        elif self.position_encoding_type == "rotary":
            # RoPE - no position embeddings added to token embeddings
            x = token_embeddings
            cos = self.cos[:T, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, T, head_dim//2)
            sin = self.sin[:T, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, T, head_dim//2)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, cos=cos, sin=sin)
            
        x = self.ln_f(x)
        return self.head(x)