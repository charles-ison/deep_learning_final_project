import torch.nn as nn
from modules.positional_encoder import PositionalEncoder


# transformer
class AudioTransformer(nn.Module):
    def __init__(self, encoder_input_size, decoder_input_size, dim_val, hidden_dim, num_layers, num_heads, dropout):
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

        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout
            )

        self.transformer = nn.Transformer(d_model=dim_val, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=hidden_dim,
                                          dropout=dropout, batch_first=True, norm_first=True)

    def forward(self, src, tgt):
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        tgt = self.decoder_input_layer(tgt)
        tgt = self.positional_encoding_layer(tgt)

        transformer_output = self.transformer(src, tgt)
        output = self.linear_mapping(transformer_output)

        return output