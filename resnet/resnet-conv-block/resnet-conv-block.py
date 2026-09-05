import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns the projection residual-block output as a nested list.
    """
    x = np.asarray(x)
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    Ws = np.asarray(Ws)
    
    hidden1 = np.clip(
        x @ W1, 
        min=0,
        max=None)
    
    y = np.clip(
        hidden1 @ W2 + x @ Ws,
        a_min=0,
        a_max=None
        )

    return np.round(y, decimals=4).tolist()