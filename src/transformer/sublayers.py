import torch.nn as nn
from .attention import MultiHeadAttention

class AttentionSubLayer(nn.Module):
    def __init__(self, num_heads, hidden_size, key_size, value_size, dropout):
        super().__init__()

        self.multihead_attention = MultiHeadAttention(
            num_heads, hidden_size, key_size, value_size
        )
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, X_Q, X_KV, key_padding_mask=None):
        attention_output = self.multihead_attention(X_Q, X_KV, key_padding_mask)
        return self.layernorm(self.dropout(attention_output) + X_Q)


class FeedForwardSubLayer(nn.Module):
    def __init__(self, hidden_size, feedforward_size, dropout):
        super().__init__()

        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, feedforward_size),
            nn.ReLU(),
            nn.Linear(feedforward_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, X):
        return self.layernorm(self.dropout(self.feedforward(X)) + X)