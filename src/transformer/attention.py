import torch
import torch.nn as nn
from .utils import slice_vertically, unslice_vertically

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_size, key_size, value_size):
        super().__init__()

        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size

        self.W_Q = nn.Parameter(torch.empty(hidden_size, key_size * num_heads))
        self.W_K = nn.Parameter(torch.empty(hidden_size, key_size * num_heads))
        self.W_V = nn.Parameter(torch.empty(hidden_size, value_size * num_heads))
        self.W_O = nn.Parameter(torch.empty(value_size * num_heads, hidden_size))

        for param in self.parameters():
            nn.init.xavier_normal_(param)

    def forward(self, X_Q, X_KV, key_padding_mask=None):
        Q = X_Q @ self.W_Q
        K = X_KV @ self.W_K
        V = X_KV @ self.W_V

        Q = slice_vertically(Q, self.key_size)
        K = slice_vertically(K, self.key_size)
        V = slice_vertically(V, self.value_size)

        A = compute_attention_matrix(Q, K, key_padding_mask=key_padding_mask)

        Y_prime = A @ V
        Y = unslice_vertically(Y_prime) @ self.W_O

        return Y

def compute_attention_matrix(Q, K, tgt_mask=None, key_padding_mask=None):
    
    E = Q @ K.transpose(-1, -2)
    
    if key_padding_mask is not None:
        mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        E = E.masked_fill(mask, float('-inf'))
    
    A = torch.softmax(E / (Q.shape[-1] ** 0.5), dim=-1)
    return A
