import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply softmax to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")

        Z_max = np.max(Z, axis=self.dim, keepdims=True)
        EZ = np.exp(Z - Z_max)
        self.A = EZ / np.sum(EZ, axis=self.dim, keepdims=True)

        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        orig_shape = dLdA.shape
        A = np.moveaxis(self.A, self.dim, -1)
        g = np.moveaxis(dLdA, self.dim, -1)
        C = g.shape[-1]
        A2 = A.reshape(-1, C)
        g2 = g.reshape(-1, C)
        # dLdZ = a * (g - (a·g) * 1) along softmax axis (Jacobian @ g in row layout)
        dot = np.sum(A2 * g2, axis=-1, keepdims=True)
        dLdZ = A2 * (g2 - dot)
        dLdZ = dLdZ.reshape(*A.shape[:-1], C)
        return np.moveaxis(dLdZ, -1, self.dim).reshape(orig_shape)
