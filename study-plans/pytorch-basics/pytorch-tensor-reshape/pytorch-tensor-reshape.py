import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    x = torch.tensor(x, dtype=float)
    match op:
        case "flatten":
            return torch.flatten(x)
        case "squeeze":
            return torch.squeeze(x)
        case "transpose":
            return x.transpose(1, 0)
