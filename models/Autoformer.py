import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len        # 96
        self.label_len = configs.label_len    # 48（解码器的开头输入部分，其余输入应该是掩码）
        self.pred_len = configs.pred_len      # 96
        self.output_attention = configs.output_attention  # false

        # Decomp  对分解模块进行初始化
        kernel_size = configs.moving_avg  # 25
        self.decomp = series_decomp(kernel_size)

        # Embedding  词嵌入：给编码器和解码器的输入作词嵌入（包含一维卷积和线性层）
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # Encoder
        '''
        编码器的参数：
            一个数组：里面包含 configs.e_layers 个编码器层EncoderLayer
                    每个EncoderLayer里面有两个一维卷积，两个序列分解模块以及一个dropout和一个relu激活函数
            一个正则化的层：用于正则化
        执行顺序:AutoCorrelation(__init__) ==>AutoCorrelationLayer(__init__) ==>EncoderLayer(__init__) ==>Encoder(__init__)
        '''
        self.encoder = Encoder(  # 给编码器配置两个EncoderLayer和一个正则化层
            [
                EncoderLayer(  # Autoformer_EncDec.py里面的一个类，EncoderLayer由一个自相关模块、两个序列分解模块和一个线性层组成
                    AutoCorrelationLayer(  # AutoCorrelationLayer是一个类，定义了一个AutoCorrelation,四个线性层,以及一些参数
                        # AutoCorrelation是一个类，其init主要是初始化一些参数
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)  # e_layers=2
            ],
            norm_layer=my_Layernorm(configs.d_model)  # 正则化,512
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(  # Autoformer_EncDec.py里面的一个类，EncoderLayer由两个自相关模块、三个序列分解模块和一个线性层组成
                    AutoCorrelationLayer(  # 第一个自相关模块，带掩码
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(  # 第二个自相关模块，不带掩码
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        x_enc:[32,96,321]   x_mark_enc:[32,96,4]
        x_dec:[32,144,321]  x_mark_dec:[32,144,4]
        enc_self_mask: [32,144,321]
        dec_self_mask: None
        dec_enc_mask: None
         """
        # decomp init 分解模块初始化
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
