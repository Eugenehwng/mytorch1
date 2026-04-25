from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np


class MultiHeadAttention:
    """
    Multi Head Attention
    """

    def __init__(self, embed_dim, num_heads):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.attention = ScaledDotProductAttention()
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def _split_heads(self, x):
        """(N, L, E) -> (N, H, L, d)"""
        N, L, E = x.shape
        x = x.reshape(N, L, self.num_heads, self.head_dim)
        return np.transpose(x, (0, 2, 1, 3))

    def _concat_heads(self, x):
        """(N, H, L, d) -> (N, L, E)"""
        N, H, L, d = x.shape
        x = np.transpose(x, (0, 2, 1, 3))
        return x.reshape(N, L, self.embed_dim)

    def _merge_masks(self, key_padding_mask, attn_mask):
        """
        key_padding_mask: (N, S) True = ignore key position
        attn_mask: (L, S) True = cannot attend (blocked)
        Returns (N, 1, L, S) bool or None
        """
        N, L, S = self.N, self.L, self.S
        combined = np.zeros((N, L, S), dtype=bool)
        if key_padding_mask is not None:
            km = np.asarray(key_padding_mask, dtype=bool)
            combined = combined | km[:, np.newaxis, :]
        if attn_mask is not None:
            am = np.asarray(attn_mask, dtype=bool)
            combined = combined | am[np.newaxis, :, :]
        return combined[:, np.newaxis, :, :]

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]

        q = self.q_proj.forward(query)
        k = self.k_proj.forward(key)
        v = self.v_proj.forward(value)

        q_h = self._split_heads(q)
        k_h = self._split_heads(k)
        v_h = self._split_heads(v)

        mask = None
        if key_padding_mask is not None or attn_mask is not None:
            mask = self._merge_masks(key_padding_mask, attn_mask)

        attn_outputs = self.attention.forward(q_h, k_h, v_h, mask=mask)
        attn_concat = self._concat_heads(attn_outputs)
        output = self.out_proj.forward(attn_concat)
        return output

    def backward(self, d_output):
        d_concat = self.out_proj.backward(d_output)
        N, L, _ = d_concat.shape
        d_attn = d_concat.reshape(N, L, self.num_heads, self.head_dim)
        d_attn = np.transpose(d_attn, (0, 2, 1, 3))

        d_qh, d_kh, d_vh = self.attention.backward(d_attn)

        d_q = self._concat_heads(d_qh)
        d_k = self._concat_heads(d_kh)
        d_v = self._concat_heads(d_vh)

        d_query = self.q_proj.backward(d_q)
        d_key = self.k_proj.backward(d_k)
        d_value = self.v_proj.backward(d_v)
        return d_query, d_key, d_value
