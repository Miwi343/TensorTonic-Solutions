import torch

def compute_gradient(values):
    """
    Returns: list of float gradient values dy/dx
    """
    x = torch.tensor(values, dtype=float, requires_grad=True)
    y = x**3 + 2*x
    loss = y.sum()
    loss.backward()
    return x.grad.tolist()
