import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    x, y = torch.tensor(x), torch.tensor(y)
    
    match op:
        case "add":
            return x + y
        case "multiply":
            return x * y
        case "matmul":
            return x @ y
        case "power":
            return x ** y
        case "max":
            return torch.maximum(x, y)