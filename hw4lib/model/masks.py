import torch


def PadMask(padded_input, input_lengths):
    """
    Boolean mask (N, T): True = padding position to ignore.
    """
    N, T = padded_input.shape[0], padded_input.shape[1]
    device = padded_input.device
    idx = torch.arange(T, device=device).view(1, T).expand(N, T)
    return idx >= input_lengths.to(device).view(N, 1)


def CausalMask(padded_input):
    """
    (T, T) bool: True = may not attend (strictly upper triangle).
    """
    T = padded_input.shape[1]
    device = padded_input.device
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
