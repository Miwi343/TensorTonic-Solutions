import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    pe = np.zeros((seq_length, d_model), dtype=float)
    
    positions = np.arange(seq_length, dtype=float)[:,None]
    # shape is (seq_len, 1)

    even = np.arange(start=0, stop=d_model, step=2, dtype=float)
    # shape is (d_model / 2)
    inv_freq = 1.0 / (10000 ** (even/d_model))
    # shape is (d_model / 2)
    
    pe[:, 0::2] = np.sin(positions * inv_freq)
    # broadcasting inv_freq (d_model / 2) -> (1, d_model / 2)
    # shape is (seq_len, d_model / 2)
    pe[:, 1::2] = np.cos(positions * inv_freq)
    
    return pe