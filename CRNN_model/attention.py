import math 
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioAttentionHead(nn.Module):
    """
    Non-causal self-attention head for audio classification.
    Every timestep can attend to ALL other timesteps (no mask).

    **Mathematical Operations:**
        1. K = X @ W_k  (Key projection)
        2. Q = X @ W_q  (Query projection)
        3. V = X @ W_v  (Value projection)
        4. scores = (Q @ K^T) / √d_k
        5. scores = masked_fill(scores, mask, -∞)
        6. attention_weights = softmax(scores)
        7. output = attention_weights @ V
    """
    def __init__(self, n_embeddings: int, head_size: int, dropout: float = 0.1):
        super().__init__()

        self.key   = nn.Linear(n_embeddings, head_size, bias=False)
        self.query = nn.Linear(n_embeddings, head_size, bias=False)
        self.value = nn.Linear(n_embeddings, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, n_embeddings)
        batch, seq_len, _ = x.shape

        # (batch, seq_len, head_size)
        key   = self.key(x)
        query = self.query(x)
        value = self.value(x)

        # attention scores
        # (batch, seq_len, seq_len)
        weights = query @ key.transpose(-2, -1)
        weights = weights / math.sqrt(key.shape[-1])

        # every timestep attends to all others
        weights = F.softmax(weights, dim=-1)

        self.last_weights = weights

        weights = self.dropout(weights)

        # (batch, seq_len, head_size)
        return weights @ value


class AudioMultiHeadAttention(nn.Module):
    def __init__(
            self, n_embeddings: int, n_heads: int,
            head_size: int, dropout: float = 0.1
        ):
        super().__init__()

        self.heads = nn.ModuleList([
            AudioAttentionHead(n_embeddings, head_size, dropout)
            for _ in range(n_heads)
        ])

        self.projection = nn.Linear(n_heads * head_size, n_embeddings)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, n_embeddings)
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # (batch, seq_len, n_heads*head_size)

        # store weights for visualization
        self.last_attention_weights = [
            head.last_weights.detach().cpu()
            for head in self.heads
            if hasattr(head, "last_weights")
        ]

        # (batch, seq_len, n_embeddings)
        out = self.projection(out) 
        out = self.dropout(out)
        return out