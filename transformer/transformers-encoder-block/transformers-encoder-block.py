import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    exps = np.exp(x)
    return exps / np.sum(exps, axis=axis, keepdims=True)

def multi_head_attention(q, k, v, W_q, W_k, W_v, W_o, num_heads):
        
        batch, seq_len, d_model = q.shape
        d_k = d_model // num_heads

        def split(x, W):
            projected =  (x @ W).reshape(batch, seq_len, num_heads, d_k)
            return projected.transpose(0, 2, 1, 3)
            # (batch, num_heads, seq_len, d_k)

        Q, K, V = split(q, W_q), split(k, W_k), split(v, W_v)

        attention = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
        # (batch, num_heads, seq_len, seq_len)
        out = softmax(attention) @ V
        # (batch, num_heads, seq_len, d_k)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        # (batch, num_heads, seq_len, d_k) -> (b, s, n, d) -> (batch, seq_len, d_model)
        return out @ W_o

def feed_forward(z, W1, b1, W2, b2):
        hidden = np.clip(z @ W1 + b1, a_min=0, a_max=None)
        return hidden @ W2 + b2

def layer_norm(x, gamma, beta, eps=1e-6):
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    
    normed = (x - mu) / np.sqrt(var + eps)
    return gamma * normed + beta

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """

    mha_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    z = layer_norm( x + mha_out, gamma=gamma1, beta=beta1)

    ffn_out = feed_forward(z, W1, b1, W2, b2)
    y = layer_norm(z + ffn_out, gamma=gamma2, beta=beta2)

    return y