import torch.nn as nn
from modules.positional_encoder import PositionalEncoder

# transformer
class AudioTransformerDecoder(nn.Module):
    def __init__(self, src_input_size, tgt_input_size, dim_val, hidden_dim, num_layers, num_heads, dropout):
        super(AudioTransformerDecoder, self).__init__()

        self.src_fc = nn.Linear(
            in_features=src_input_size,
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

        # NOTE: Is this redundant?
        # TODO: @Chase Investigate positional encodings.
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout
        )

        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=dim_val, nhead=num_heads, dim_feedforward=hidden_dim,
                                                               dropout=dropout, batch_first=True, norm_first=True)

        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_layers)

    def forward(self, src, tgt):

        src = self.src_fc(src)
        src = self.positional_encoding_layer(src)

        tgt = self.tgt_fc(tgt)
        tgt = self.positional_encoding_layer(tgt)

        transformer_output = self.transformer_decoder(src, tgt)
        output = self.fc_output(transformer_output)

        return output