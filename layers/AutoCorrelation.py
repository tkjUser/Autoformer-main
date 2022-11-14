import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor  # factor=3
        self.scale = scale    # None
        self.mask_flag = mask_flag  # false
        self.output_attention = output_attention  # false
        self.dropout = nn.Dropout(attention_dropout)  # 0.05

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design) 加速版Autoformer（批归一化）
        This is for the training phase.
        """
        # 把批次看成1 进行分析，一目了然
        head = values.shape[1]  # 4
        channel = values.shape[2]  # 128
        length = values.shape[3]  # 96
        # find top k
        top_k = int(self.factor * math.log(length))  # 1* ln(96) = 4
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # 32, 96 每次求mean 那个维度就会消失
        # 4 假设得到：[3, 4, 5, 2] # 这里[1]后得到的是index  ## 批次维度求均值得到 shape: 96  再取 top_k
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)  # 32 4  把96个里面最大的4个值取出来
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)  # 32 4  这4个值归一化
        # aggregation
        tmp_values = values  # 32 4 128 96
        delays_agg = torch.zeros_like(values).float()  # 32 4 128 96 都是0
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)  # 32 4 128  96  从序列那个维度移动  # index 对应移动的步长
            # pattern 乘的是 那个 R_q,k 后面部分表示对应元素相乘 后面的就是那个权重  unsqueeze repeat 是为了形状相同 可以相乘
            # tmp_corr[:, i] 这里不考虑batch_size 时是一个数 把这个数repeat 96次
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0) \
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0) \
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)  # 32, 4, 128, 49
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)  # 32, 4, 128, 49
        res = q_fft * torch.conj(k_fft)  # 32, 4, 128, 49
        corr = torch.fft.irfft(res, dim=-1)  # 32, 4, 128, 96

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):  # 传入AutoCorrelation类的对象correlation
        super(AutoCorrelationLayer, self).__init__()
        # 多头自相关也是输入512维，有8个头，每个头的k和v都是64
        d_keys = d_keys or (d_model // n_heads)      # d_keys = 64
        d_values = d_values or (d_model // n_heads)  # d_values = 64

        self.inner_correlation = correlation  # AutoCorrelation((dropout): Dropout(p=0.05, inplace=False))
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)  # 将Q映射 Linear(in_features=512, out_features=512, bias=True)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)    # 将K映射 Linear(in_features=512, out_features=512, bias=True)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)  # 将V映射 Linear(in_features=512, out_features=512, bias=True)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)    # 输出值时所有头（head）的综合 Linear(in_features=512, out_features=512, bias=True)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
