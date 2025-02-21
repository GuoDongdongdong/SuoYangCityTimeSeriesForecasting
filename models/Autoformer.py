import torch
import torch.nn as nn

from .layers.Embedding import DataEmbedding_wo_pos
from .layers.AutoformerDetail import Encoder, EncoderLayer, Decoder, DecoderLayer, series_decomp, AutoCorrelation, AutoCorrelationLayer, my_Layernorm
from .layers.RevIN import RevIN
from models.BaseForecastModel import BaseForecastModel
from utils.ExperimentArgs import ExperimentArgs
from utils.functions import calc_mse


class Model(BaseForecastModel):
    def __init__(self, exp_args:ExperimentArgs):
        super().__init__()
        self.predict_length = exp_args['predict_length']
        self.label_length = exp_args['label_length']
        self.dropout = exp_args['dropout']
        self.moving_avg = exp_args['moving_avg']
        self.enc_in = len(exp_args['targets'])
        self.dec_in = len(exp_args['targets'])
        self.c_out = len(exp_args['targets'])
        self.d_model = exp_args['d_model']
        self.embed = exp_args['embed']
        self.freq = exp_args['date_frequence']
        self.factor = exp_args['factor']
        self.n_heads = exp_args['n_heads']
        self.e_layers = exp_args['e_layers']
        self.d_layers = exp_args['d_layers']
        self.d_ff = exp_args['d_ff']
        self.activation = exp_args['activation']
        self.output_attention = exp_args['output_attention']
        self.subtract_last = exp_args['subtract_last']
        self.affine = exp_args['affine']
        self.revin = exp_args['revin']
        self.model = _Autoformer(
            self.predict_length,
            self.label_length,
            self.dropout,
            self.moving_avg,
            self.enc_in,
            self.dec_in,
            self.c_out,
            self.d_model,
            self.embed,
            self.freq,
            self.factor,
            self.n_heads,
            self.e_layers,
            self.d_layers,
            self.d_ff,
            self.activation,
            self.output_attention,
            self.subtract_last,
            self.affine,
            self.revin,
        )

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
    

class _Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, 
                 predict_len:int,
                 label_len:int,
                 dropout:float,
                 moving_avg:int,
                 enc_in:int,
                 dec_in:int,
                 c_out:int,
                 d_model:int,
                 embed:str,
                 freq:str,
                 factor:int,
                 n_heads:int,
                 e_layers:int,
                 d_layers:int,
                 d_ff:int,
                 activation:str,
                 output_attention:int,
                 subtract_last:int,
                 affine:int,
                 revin:int,
                 ):
        super(_Autoformer, self).__init__()
        self.predict_len = predict_len
        self.label_len = label_len
        self.output_attention = output_attention
        # Decomp
        self.decomp = series_decomp(moving_avg)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq,
                                                  dropout)
        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq,
                                                  dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

        if revin:
            self.revin_layer = RevIN(enc_in, affine=affine, subtract_last=subtract_last)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.predict_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.predict_len, x_dec.shape[2]], device=x_enc.device)
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
        return dec_out[:, -self.predict_len:, :]

