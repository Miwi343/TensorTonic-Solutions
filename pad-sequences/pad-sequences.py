import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    n = len(seqs)
    
    if max_len is None:
        max_len = max([len(row) for row in seqs], default=0)

    out = np.full((n, max_len), pad_value, dtype=int)
    for i, s in enumerate(seqs):
        k = min(len(s), max_len)
        out[i,:k] = s[:k]

    return out
    