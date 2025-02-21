import torch
import torch.nn as nn
from .layers.TransformerDetail import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .layers.SelfAttention import ProbAttention, AttentionLayer
from .layers.Embedding import DataEmbedding

from models.BaseForecastModel import BaseForecastModel
from utils.ExperimentArgs import ExperimentArgs
from utils.functions import calc_mse


class Model(BaseForecastModel):
    def __init__(self, exp_args:ExperimentArgs):
        super().__init__()
        self.model = _Informer(exp_args)
    
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
    

class _Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, exp_args:ExperimentArgs):
        super(_Informer, self).__init__()
        self.enc_in = len(exp_args['targets'])
        self.dec_in = len(exp_args['targets'])
        self.c_out = len(exp_args['targets'])
        self.pred_len = exp_args['predict_length']
        self.freq = exp_args['date_frequence']
        self.dropout = exp_args['dropout']
        self.embed = exp_args['embed']
        self.distil = exp_args['distil']
        self.d_model = exp_args['d_model']
        self.factor = exp_args['factor']
        self.n_heads = exp_args['n_heads']
        self.e_layers = exp_args['e_layers']
        self.d_layers = exp_args['d_layers']
        self.d_ff = exp_args['d_ff']
        self.activation = exp_args['activation']
        self.output_attention = exp_args['output_attention']

        # Embedding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq,
                                           self.dropout)
        self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq,
                                           self.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            [
                ConvLayer(
                    self.d_model
                ) for l in range(self.e_layers - 1)
            ] if self.distil else None,
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    AttentionLayer(
                        ProbAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
