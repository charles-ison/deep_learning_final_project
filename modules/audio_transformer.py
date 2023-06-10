import torch.nn as nn
from modules.positional_encoding import PositionalEncoding

class AudioTransformer(nn.Module):
    def __init__(self, encoder_input_size, decoder_input_size, max_len, dim_val, hidden_dim, num_layers, num_heads, dropout):
        super(AudioTransformer, self).__init__()

        self.encoder_input_layer = nn.Linear(
            in_features=encoder_input_size, 
            out_features=dim_val 
        )

        self.decoder_input_layer = nn.Linear(
            in_features=decoder_input_size,
            out_features=dim_val
        )
        
        self.linear_mapping = nn.Linear(
            in_features=dim_val, 
            out_features=decoder_input_size
        )

        self.positional_encoding = PositionalEncoding(d_model=dim_val, dropout=dropout, max_len=max_len)

        self.transformer = nn.Transformer(d_model=dim_val, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=hidden_dim,
                                          dropout=dropout, batch_first=True, norm_first=True)

    def forward(self, src, tgt):
        src = self.encoder_input_layer(src)
        src = self.positional_encoding(src)
        tgt = self.decoder_input_layer(tgt)
        tgt = self.positional_encoding(tgt)

        transformer_output = self.transformer(src, tgt)
        output = self.linear_mapping(transformer_output)

        return output