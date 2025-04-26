import torch

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    print("MPS is available and built.")
else:
    print("MPS is not available.")