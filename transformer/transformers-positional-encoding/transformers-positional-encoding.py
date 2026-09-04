import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    pe = np.zeros((seq_length, d_model), dtype=float)
    
    positions = np.arange(seq_length, dtype=float)[:,None]

    even = np.arange(start=0, stop=d_model, step=2, dtype=float)

    inv_freq = 1.0 / (10000 ** (even/d_model))
    
    pe[:, 0::2] = np.sin(positions * inv_freq)
    pe[:, 1::2] = np.cos(positions * inv_freq)
    
    return pe