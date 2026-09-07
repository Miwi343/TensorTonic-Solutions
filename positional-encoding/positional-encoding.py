import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    pe = np.zeros((seq_len, d_model))

    positions = np.arange(start=0, stop=seq_len)[:,None]
    # need to make the 2i column matrices
    even_indeces = np.arange(start=0, stop=d_model, step=2)
    inv_freq = 1 / (base ** (even_indeces / d_model))

    angles = positions * inv_freq
    
    pe[: ,0::2] = np.sin( angles )
    pe[:, 1::2] = np.cos( angles[:, :d_model//2])

    return pe