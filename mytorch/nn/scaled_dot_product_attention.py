import numpy as np
from .activation import Softmax


class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """

    def __init__(self):
        self.eps = 1e10  # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        """
        :param Q: (N, ..., H, L, E)
        :param K: (N, ..., H, S, E)
        :param V: (N, ..., H, S, Ev)
        :param mask: bool, True = masked (ignore), broadcastable to (..., L, S)
        :return: (N, ..., H, L, Ev)
        """
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask
        self.dk = float(Q.shape[-1])

        scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(self.dk)
        if mask is not None:
            scores = np.where(mask, -self.eps, scores)

        self.attention_scores = self.softmax.forward(scores)
        output = np.matmul(self.attention_scores, V)
        return output

    def backward(self, d_output):
        """
        :param d_output: (N, ..., H, L, Ev)
        :return: dQ, dK, dV same leading dims as Q,K,V
        """
        P = self.attention_scores
        d_V = np.matmul(np.swapaxes(P, -1, -2), d_output)
        d_P = np.matmul(d_output, np.swapaxes(self.V, -1, -2))
        d_scores = self.softmax.backward(d_P)
        if self.mask is not None:
            d_scores = np.where(self.mask, 0.0, d_scores)

        d_scores_scaled = d_scores / np.sqrt(self.dk)
        d_Q = np.matmul(d_scores_scaled, self.K)
        d_K = np.matmul(np.swapaxes(d_scores_scaled, -1, -2), self.Q)
        return d_Q, d_K, d_V
