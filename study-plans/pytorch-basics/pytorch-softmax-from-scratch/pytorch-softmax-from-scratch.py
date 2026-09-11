import torch

def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    x = logits - logits.amax(dim=-1, keepdim=True)
    exps = torch.exp(x)

    return exps / exps.sum(dim=-1, keepdim=True)
