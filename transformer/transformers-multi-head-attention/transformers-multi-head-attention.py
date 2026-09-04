import numpy as np

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    # calculate shape
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    def split(x, W):
        projected = (x @ W).reshape(batch, seq_len, num_heads, d_k)
        return projected.transpose(0, 2, 1, 3)
    
    q = split(Q, W_q)
    k = split(K, W_k)
    v = split(V, W_v)

    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_k)

    def softmax( x, axis=-1 ):
        x = x - x.max(axis=axis, keepdims=True)
        exps = np.exp(x)
        return exps / exps.sum( axis = axis, keepdims=True)
    
    out = softmax(scores) @ v
    out = out.transpose(0, 2, 1, 3)
    out = out.reshape(batch, seq_len, d_model) 
    return out @ W_o