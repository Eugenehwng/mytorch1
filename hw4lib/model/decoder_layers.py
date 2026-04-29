import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, CrossAttentionLayer, FeedForwardLayer


class SelfAttentionDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout)
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, weights = self.self_attn(
            x, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=need_attn_weights
        )
        x = self.ffn(x)
        return x, weights


class CrossAttentionDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout)
        self.cross_attn = CrossAttentionLayer(d_model, num_heads, dropout)
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        dec_key_padding_mask: Optional[torch.Tensor] = None,
        enc_key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x, self_w = self.self_attn(
            x,
            key_padding_mask=dec_key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_attn_weights,
        )
        x, cross_w = self.cross_attn(
            x,
            enc_output,
            key_padding_mask=enc_key_padding_mask,
            attn_mask=None,
            need_weights=need_attn_weights,
        )
        x = self.ffn(x)
        return x, self_w, cross_w
