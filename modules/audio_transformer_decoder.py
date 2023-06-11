import torch
import torch.nn as nn
from modules.positional_encoding import PositionalEncoding

class AudioTransformerDecoder(nn.Module):
    def __init__(self, mem_input_size, tgt_input_size, max_len, dim_val, hidden_dim, num_layers, num_heads, dropout):
        super(AudioTransformerDecoder, self).__init__()

        self.max_len = max_len
        self.start_token = nn.Parameter(torch.randn(1, 1, dim_val))

        self.mem_fc = nn.Linear(
            in_features=mem_input_size,
            out_features=dim_val
        )

        self.tgt_fc = nn.Linear(
            in_features=tgt_input_size,
            out_features=dim_val
        )
        
        self.fc_output = nn.Linear(
            in_features=dim_val, 
            out_features=tgt_input_size
        )

        self.positional_encoding = PositionalEncoding(d_model=dim_val, dropout=dropout, max_len=max_len)

        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=dim_val, nhead=num_heads, dim_feedforward=hidden_dim,
                                                               dropout=dropout, batch_first=True, norm_first=True)

        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_layers)

    def forward(self, mem, tgt, tgt_mask=None):

        tgt = self.tgt_fc(tgt)
        tgt = self.positional_encoding(tgt)
        start_tokens = self.start_token.repeat(tgt.shape[0], 1, 1)
        tgt = torch.cat((start_tokens, tgt), dim = 1)

        mem = self.mem_fc(mem)
        mem = self.positional_encoding(mem)
        mem = torch.cat((start_tokens, mem), dim=1)

        # Note, the decoder switches the order these need to be passed from the vanilla transformer (encoder and decoder)
        transformer_output = self.transformer_decoder(tgt, mem, tgt_mask)
        output = self.fc_output(transformer_output)

        # Removing end token
        output = output[:, 0:self.max_len, :]

        return output