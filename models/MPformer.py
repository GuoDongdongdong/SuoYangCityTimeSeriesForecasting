from typing import Optional

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from models.BaseForecastModel import BaseForecastModel
from utils.ExperimentArgs import ExperimentArgs
from utils.functions import calc_mse


class Model(BaseForecastModel):
    def __init__(self, exp_args:ExperimentArgs):
        super().__init__()
        self.lookback_length = exp_args['lookback_length']
        self.predict_length = exp_args['predict_length']
        self.dropout = exp_args['dropout']
        self.e_layers = exp_args['e_layers']
        self.n_heads = exp_args['n_heads']
        self.d_model = exp_args['d_model']
        self.d_ff = exp_args['d_ff']
        self.activate_type = exp_args['activate_type']
        self.norm_type = exp_args['norm_type']
        self.patch_length = exp_args['patch_length']
        self.patch_stride = exp_args['patch_stride']
        self.model = _MPformer(
            self.lookback_length,
            self.predict_length,
            self.dropout,
            self.e_layers,
            self.n_heads,
            self.d_model,
            self.d_ff,
            self.activate_type,
            self.norm_type,
            self.patch_length,
            self.patch_stride
        )

    def evaluate(self, batch, training):
        x = batch['observed_data']
        y = self.model.forward(x)
        return calc_mse(x, y)
    
    def forecast(self, batch):
        x = batch['observed_data']
        y = self.model.forward(x)
        return y


class _MPformer(nn.Module):
    def __init__(self,
                 lookback_length:int,
                 predict_length:int,
                 dropout:float,
                 e_layers:int,
                 n_heads:int,
                 d_model:int,
                 d_ff:int,
                 activate_type:str,
                 norm_type:str,
                 patch_length:int,
                 patch_stride:int,
                ):
        super(_MPformer, self).__init__()
        self.dimension = lookback_length // (patch_length - 1)
        self.patch_num = (lookback_length - patch_length + patch_stride) // patch_stride

        self.patch = MultiPatch(lookback_length, patch_length, patch_stride)
        self.project = nn.Sequential(
            nn.Linear(patch_length, d_model),
            nn.Dropout(dropout),
        )

        # encoder
        self.encoder = Encoder(
            e_layers=e_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activate_type=activate_type,
            norm_type=norm_type
        )

        # flatten linear
        self.flatten_linear_head = nn.Sequential(
            nn.Flatten(start_dim=-3),
            nn.Linear(d_model * self.patch_num * self.dimension, predict_length),
        )

    # x [B, L, D = 1]
    def forward(self, x):
        # patching
        # x [B, channel, patch_num, patch_len]
        x = self.patch(x)

        # project
        # x [B, channel, patch_num, d_model]
        x = self.project(x)
        b, d, patch_num, d_model = x.shape
        x = torch.reshape(x, (b * d, patch_num, d_model))

        # encoder
        x = self.encoder(x)
        # x [B, channel, patch_num, d_model]
        x = torch.reshape(x, (b, d, patch_num, d_model))

        # flatten
        # output [B, predict_len]
        output = self.flatten_linear_head(x)
        # output [B, 1, predict_len]
        output = torch.unsqueeze(output, 1)
        # output [B, predict_len, 1]
        output = output.permute(0, 2, 1)
        return output


class MultiPatch(nn.Module):
    def __init__(self, lookback_len, patch_len, patch_stride):
        super(MultiPatch, self).__init__()

        self.dimension = lookback_len // (patch_len - 1)
        self.patch_num = (lookback_len - patch_len + patch_stride) // patch_stride
        self.paddings = nn.ModuleList()
        self.unfolds = nn.ModuleList()
        for dilation in range(1, self.dimension + 1):
            padding_len = (dilation - 1) * (patch_len - 1)
            padding_l = padding_len // 2
            padding_r = padding_len - padding_l
            self.paddings.append(nn.ReplicationPad1d(padding=(padding_l, padding_r)))
            self.unfolds.append(nn.Unfold(kernel_size=(1, patch_len), dilation=dilation, padding=0, stride=patch_stride))

    # x [B, L, D = 1]
    def forward(self, x):
        output = []
        # x [B, D = 1, L]
        x = x.permute(0, 2, 1)
        for _ in range(self.dimension):
            y = self.paddings[_](x)
            y = torch.unsqueeze(y, dim=1)
            # y [B, Patch_len, Patch_num]
            y = self.unfolds[_](y)
            # y [B, Patch_num, Patch_len]
            y = y.permute(0, 2, 1)
            output.append(y)
        # output [B, dimension(channel), patch_num, patch_len]
        output = torch.stack(output, dim=1)
        return output


class Encoder(nn.Module):
    def __init__(self, e_layers, n_heads, d_model, d_ff, dropout, activate_type, norm_type):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList([EncoderLayer(
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activate_type=activate_type,
            norm_type=norm_type,
        ) for _ in range(e_layers)])

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, dropout, activate_type, norm_type):
        super(EncoderLayer, self).__init__()
        self.mutil_attention = _MultiheadAttention(d_model, n_heads)
        self.attention_dropout = nn.Dropout(dropout)

        if norm_type == 'batch':
            self.attention_norm = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.attention_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation_function(activate_type),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)

        if norm_type == 'batch':
            self.ffn_norm = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        y, attention = self.mutil_attention(x, x, x)
        y = x + self.attention_dropout(y)
        y = self.attention_norm(y)

        z = self.ffn(y)
        z = y + self.ffn_dropout(z)
        z = self.ffn_norm(z)
        return z


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0.,
                 proj_dropout=0.,
                 qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                   res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None,
                prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,
                                                                         2)  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,
                                                                       1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,
                                                                         2)  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask,
                                                              attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask,
                                                 attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


def activation_function(activate_type):
    activation_dict = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
    }
    return activation_dict[activate_type]


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomposition(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
