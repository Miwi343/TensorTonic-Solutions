import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype=float)
    result = None
    match method:
        case "relu":
            result = torch.clamp(x, min=0)
        case "sigmoid":
            result = 1 / (1 + torch.exp(-1*x))
        case "tanh":
            result = torch.tanh(x)
        case "leaky_relu":
            result = torch.where(x > 0, x, 0.01*x)

    if result is not None:
        return result.tolist()

    return None