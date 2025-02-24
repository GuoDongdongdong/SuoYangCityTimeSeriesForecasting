import torch
import torch.nn as nn

from .layers.TransformerDetail import Encoder, EncoderLayer
from .layers.SelfAttention import ReformerLayer
from .layers.Embedding import DataEmbedding
from models.BaseForecastModel import BaseForecastModel
from utils.ExperimentArgs import ExperimentArgs
from utils.functions import calc_mse


class Model(BaseForecastModel):
    def __init__(self, exp_args:ExperimentArgs):
        super().__init__()
        self.model = _Reformer(exp_args)
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
    

class _Reformer(nn.Module):
    """
    Reformer with O(LlogL) complexity
    - It is notable that Reformer is not proposed for time series forecasting, in that it cannot accomplish the cross attention.
    - Here is only one adaption in BERT-style, other possible implementations can also be acceptable.
    - The hyper-parameters, such as bucket_size and n_hashes, need to be further tuned.
    The official repo of Reformer (https://github.com/lucidrains/reformer-pytorch) can be very helpful, if you have any questiones.
    """

    def __init__(self, exp_args:ExperimentArgs):
        super(_Reformer, self).__init__()
        self.enc_in = len(exp_args['targets'])
        self.c_out = len(exp_args['targets'])
        self.freq = exp_args['date_frequence']
        self.dropout = exp_args['dropout']
        self.pred_len = exp_args['predict_length']
        self.d_model = exp_args['d_model']
        self.embed = exp_args['embed']
        self.n_heads = exp_args['n_heads']
        self.bucket_size = exp_args['bucket_size']
        self.n_hashes = exp_args['n_hashes']
        self.d_ff = exp_args['d_ff']
        self.activation = exp_args['activation']
        self.e_layers = exp_args['e_layers']
        self.d_layers = exp_args['d_layers']
        self.output_attention = exp_args['output_attention']

        # Embedding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq,
                                           self.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, self.d_model, self.n_heads, bucket_size=self.bucket_size,
                                  n_hashes=self.n_hashes),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # add placeholder
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
        # Reformer: encoder only
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out[:, -self.pred_len:, :], attns
        else:
            return enc_out[:, -self.pred_len:, :]  # [B, L, D]
