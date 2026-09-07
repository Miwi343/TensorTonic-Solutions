import torch

def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """

    X = torch.tensor(X)
    W = torch.tensor(W)
    b = torch.tensor(b)

    Y = X @ W + b
    return Y.tolist()