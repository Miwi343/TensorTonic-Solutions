import numpy as np

def identity_block(x, W1, W2):
    """
    Returns the identity residual-block output as a nested list.
    """
    x = np.array(x)
    W1, W2 = np.asarray(W1), np.asarray(W2)
    
    hidden = np.clip(x @ W1.transpose(1,0), a_min=0, a_max=None)
    y = np.clip(hidden @ W2.transpose(1,0) + x, a_min=0, a_max=None)
    
    return np.round(y, decimals=4).tolist()