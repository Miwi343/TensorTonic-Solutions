import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    scores = np.asarray(scores)
    T = scores.shape[-1]
    
    mask = np.tril(np.ones((T,T), dtype=bool), k=0)

    result = np.where(mask, scores, mask_value)
    return result