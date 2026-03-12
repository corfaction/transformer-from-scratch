import sys
sys.path.append('../cpp_module')

from transformer.encoder import Encoder
import torch
import tokenizer

t = tokenizer.Tokenizer()
t.load("../artifacts")

config = {
    "stack_size": 6,
    "num_heads": 8, 
    "hidden_size": 512, 
    "key_size": 64,    
    "value_size": 64,  
    "feedforward_size": 2048, 
    "dropout": 0.1            
}

encoder = Encoder(
    transformer_encoder_config=config,
    vocab_size=t.getVocabSize(),
    context_size=512
)

tokens = t.encode("Input Text")
X = torch.tensor(tokens).unsqueeze(0)

output = encoder(X)
print(f"Output shape: {output.shape}")
print(f"Output: {output}")