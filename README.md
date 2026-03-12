# Transformer from Scratch

Implementation of Transformer architecture from scratch, following "Transformer Tutorial with PyTorch". Includes custom BPE tokenizer in C++ with Python bindings.

## ✨ Features

- **Custom BPE Tokenizer** (C++/pybind11)
  - Byte-level BPE training from scratch
  - Save/load vocabulary and merges (JSON + text format)
  - Python bindings for easy integration
  - Telegram chat parser for dataset creation

- **Transformer Components** (PyTorch)
  - Token embedding + positional encoding
  - Multi-head self-attention with learnable weight matrices
  - Encoder stack with Add & Norm sublayers
  - Feed-forward networks with ReLU activation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch
- CMake 3.10+
- C++17 compiler
- pybind11
- nlohmann/json

## Installation

### Clone the repository:
   
    git clone https://github.com/corfaction/transformer-from-scratch.git
    cd transformer-from-scratch

    pip install torch pybind11

    Download and place nlohmann/json.hpp:
        - Download json.hpp from the nlohmann/json releases
        - Place it in a known location (e.g., C:/Users/LocalAdmin/libs/)

    Build the tokenizer:

    cd src/tokenizer
    mkdir build && cd build
    cmake -DCMAKE_PREFIX_PATH=path/to/pybind11 -DJSON_INCLUDE_DIR=path/to/json.hpp/dir ..
    cmake --build . --config Release

    Note: Copy the built tokenizer.pyd (Windows) or tokenizer.so (Linux) to your Python path or set sys.path.append('../cpp_module')

### Tokenizer Example
    python

    import sys
    sys.path.append('../cpp_module')  # or wherever your compiled module is
    import tokenizer

### Create and train tokenizer
    tokenizer = tokenizer.Tokenizer()
    tokenizer.train("path/to/dataset", 5000)  # train to 5000 tokens
    tokenizer.save("path/to/folder/artifacts")

### Encode/decode
    encoded = new_tokenizer.encode("hello world")
    decoded = new_tokenizer.decode(encoded)
    print(f"Vocabulary size: {new_tokenizer.getVocabSize()}")

    Transformer Example
    python

    import sys
    sys.path.append('../cpp_module')
    from transformer.encoder import Encoder
    import torch
    import tokenizer

### Load tokenizer
    t = tokenizer.Tokenizer()
    t.load("../artifacts")

### Configure transformer
    config = {
        "stack_size": 6,
        "num_heads": 8, 
        "hidden_size": 512, 
        "key_size": 64,    
        "value_size": 64,  
        "feedforward_size": 2048, 
        "dropout": 0.1            
    }

### Create encoder
    encoder = Encoder(
        transformer_encoder_config=config,
        vocab_size=t.getVocabSize(),
        context_size=512
    )

### Forward pass
    tokens = t.encode("Input Text")
    X = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension
    output = encoder(X)
    print(f"Output shape: {output.shape}")

📚 Based On

    - "Transformer Tutorial with PyTorch"
    - "Attention Is All You Need" paper
    -  BPE tokenization (Byte Pair Encoding)

📝 License

MIT License - see LICENSE file
👤 Author: corfaction