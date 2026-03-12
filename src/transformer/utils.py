import torch

def slice_vertically(X, slice_size):
    return X.unflatten(dim=-1, sizes=(-1, slice_size)).transpose(-2, -3)

def unslice_vertically(X):
    return X.transpose(-2, -3).flatten(-2, -1)