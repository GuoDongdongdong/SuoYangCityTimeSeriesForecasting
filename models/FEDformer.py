import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Embedding import DataEmbedding_wo_pos
from .layers.AutoformerDetail import AutoCorrelationLayer
from .layers.FEDformerDetail import FourierBlock, FourierCrossAttention
from .layers.FEDformerDetail import MultiWaveletCross, MultiWaveletTransform
from .layers.AutoformerDetail import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi

from models.BaseForecastModel import BaseForecastModel
from utils.ExperimentArgs import ExperimentArgs
from utils.functions import calc_mse


class Model(BaseForecastModel):
    def __init__(self, exp_args:ExperimentArgs):
        super().__init__()
        self.model = _FEDformer(exp_args)
        self.label_length = exp_args['label_length']
        
    def _get_inputs(self, batch:dict) -> dict:
        observed_data = batch['observed_data']
        predict_data = batch['predict_data']
        observed_date = batch['observed_date']
        predict_date = batch['predict_date']
        device = observed_data.device
        decoder_data = torch.zeros_like(predict_data).float()
        decoder_data = torch.cat([observed_data[:, -self.label_length:, :], decoder_data], dim=1).float().to(device)
        decoder_date = torch.cat([observed_date[:, -self.label_length:, :], predict_date], dim=1).float().to(device)
        inputs = {
            'observed_data' : observed_data,
            'observed_date' : observed_date,
            'decoder_data'  : decoder_data,
            'decoder_date'  : decoder_date,
        }
        return inputs
    
    def evaluate(self, batch:dict, training:bool) -> torch.Tensor:
        inputs = self._get_inputs(batch)
        x_enc = inputs['observed_data']
        x_mark_enc = inputs['observed_date']
        x_dec = inputs['decoder_data']
        x_mark_dec = inputs['decoder_date']
        predict = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        y = batch['predict_data']
        return calc_mse(y, predict)
    
    def forecast(self, batch) -> torch.Tensor:
        inputs = self._get_inputs(batch)
        x_enc = inputs['observed_data']
        x_mark_enc = inputs['observed_date']
        x_dec = inputs['decoder_data']
        x_mark_dec = inputs['decoder_date']
        predict = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return predict
    
    

class _FEDformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, exp_args:ExperimentArgs):
        super(_FEDformer, self).__init__()
        self.device = self._get_device(exp_args)
        self.seq_len = exp_args['lookback_length']
        self.label_len = exp_args['label_length']
        self.pred_len = exp_args['predict_length']
        self.enc_in = len(exp_args['targets'])
        self.dec_in = len(exp_args['targets'])
        self.c_out = len(exp_args['targets'])
        self.freq = exp_args['date_frequence']
        self.dropout = exp_args['dropout']

        self.modes = exp_args['modes']
        self.version = exp_args['version']
        self.mode_select = exp_args['mode_select']
        self.L = exp_args['l']
        self.base = exp_args['base']
        self.cross_activation = exp_args['cross_activation']
        self.embed = exp_args['embed']
        self.d_model = exp_args['d_model']
        self.d_ff = exp_args['d_ff']
        self.n_heads = exp_args['n_heads']
        self.e_layers = exp_args['e_layers']
        self.d_layers = exp_args['d_layers']
        self.activation = exp_args['activation']
        self.moving_avg = exp_args['moving_avg']
        self.output_attention = exp_args['output_attention']

        # Decomp
        kernel_size = self.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.d_model, self.embed, self.freq,
                                                  self.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(self.dec_in, self.d_model, self.embed, self.freq,
                                                  self.dropout)

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=self.d_model, L=self.L, base=self.base)
            decoder_self_att = MultiWaveletTransform(ich=self.d_model, L=self.L, base=self.base)
            decoder_cross_att = MultiWaveletCross(in_channels=self.d_model,
                                                  out_channels=self.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=self.modes,
                                                  ich=self.d_model,
                                                  base=self.base,
                                                  activation=self.cross_activation)
        else:
            encoder_self_att = FourierBlock(in_channels=self.d_model,
                                            out_channels=self.d_model,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_self_att = FourierBlock(in_channels=self.d_model,
                                            out_channels=self.d_model,
                                            seq_len=self.seq_len//2+self.pred_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=self.d_model,
                                                      out_channels=self.d_model,
                                                      seq_len_q=self.seq_len//2+self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=self.modes,
                                                      mode_select_method=self.mode_select)
        # Encoder
        enc_modes = int(min(self.modes, self.seq_len//2))
        dec_modes = int(min(self.modes, (self.seq_len//2+self.pred_len)//2))
        logging.debug('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        self.d_model, self.n_heads),

                    self.d_model,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        self.d_model, self.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.c_out,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=my_Layernorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )
    
    def _get_device(self, exp_args:ExperimentArgs) -> torch.device:
        if exp_args['use_gpu']:
            GPU_id = exp_args['gpu_id']
            os.environ["CUDA_VISIBLE_DEVICES"] = GPU_id
            device = torch.device(f"cuda:{GPU_id}")
        else:
            device = torch.device('cpu')
        return device

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(self.device)  # cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
