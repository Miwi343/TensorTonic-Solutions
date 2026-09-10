import torch

def batch_norm(X, gamma, beta, eps=1e-5):
    """
    Returns: tensor of shape (N, D), the batch-normalized output
    """

    mu = torch.mean(X, dim=0)
    sigma = torch.var(X, dim=0, unbiased=False)

    x_norm = ( X - mu ) / torch.sqrt(sigma + eps)

    return gamma * x_norm + beta
