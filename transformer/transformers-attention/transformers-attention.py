import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    relation = Q @ K.transpose(-2, -1)
    sqrt_d_k = torch.sqrt( torch.tensor(K.shape[-1]) )

    relation_prob = torch.softmax( relation / sqrt_d_k, dim=-1)
    return relation_prob @ V