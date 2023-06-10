import torch.nn as nn
from modules.positional_encoding import PositionalEncoding

class AudioTransformerDecoder(nn.Module):
    def __init__(self, mem_input_size, tgt_input_size, max_len, dim_val, hidden_dim, num_layers, num_heads, dropout):
        super(AudioTransformerDecoder, self).__init__()

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

    def forward(self, mem, tgt):

        tgt = self.tgt_fc(tgt)
        tgt = self.positional_encoding(tgt)

        mem = self.mem_fc(mem)
        mem = self.positional_encoding(mem)

        # TODO: At inference time this flat needs to be removed
        # Note, the decoder switches the order these need to be passed from the encoder-decoder
        transformer_output = self.transformer_decoder(tgt, mem, tgt_is_causal=True)
        output = self.fc_output(transformer_output)

        return output