import torch
import torch.nn as nn

hidden_size = 512
key_size = 512
value_size = 512

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
    def forward(self, token_indices):
        return self.embedding(token_indices)
    
emb = TokenEmbedding(5000, hidden_size)
X = emb(torch.tensor([332, 432, 51, 2156]))

torch.set_printoptions(precision=2)

WQ = torch.rand(hidden_size, key_size)
WK = torch.rand(hidden_size, key_size)
WV = torch.rand(hidden_size, key_size)

Q = X @ WQ
K = X @ WK
V = X @ WV

scores = Q @ K.T
scores = scores / (key_size ** 0.5)
weights = torch.softmax(scores, dim=1)
Y = weights @ V

print(Y)