import numpy as np

def rnn_step_forward(x_t: list, h_prev: list, Wx: list, Wh: list, b: list) -> np.ndarray:
    """
    Returns a NumPy array with shape (H,).
    """
    x_t, h_prev, Wx, Wh, b = np.asarray(x_t), np.asarray(h_prev), np.asarray(Wx), np.asarray(Wh), np.asarray(b)

    
    a_t = x_t @ Wx + h_prev @ Wh + b
    return np.tanh(a_t)
    