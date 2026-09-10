import torch

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """
    pred = torch.tensor(pred, dtype=float)
    target = torch.tensor(target)
    match method:
        case "mse":
            L = torch.mean((pred - target)**2)
        case "cross_entropy":
            n,c = pred.shape
            logs = torch.log_softmax(pred, dim=-1, dtype=float)
            L = torch.mean( -logs[torch.arange(n), target] )
        case "huber":
            a = torch.abs(pred-target)
            quadratic_case = 1/2 * a**2
            linear_case = delta*(a - delta/2)
            L = torch.mean( torch.where(a > delta, linear_case, quadratic_case) )

    return L
