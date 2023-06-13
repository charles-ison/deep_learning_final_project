import torch
import torch.nn as nn
from modules.positional_encoding import PositionalEncoding

class AudioTransformerDecoder(nn.Module):
    def __init__(self, mem_input_size, tgt_voc_size, max_len, embedding_dim, num_q, hidden_dim, num_layers, num_heads, dropout):
        super(AudioTransformerDecoder, self).__init__()
        dim_model = embedding_dim * num_q
        self.dim_model = dim_model
        self.max_len = max_len
        self.num_q = num_q
        self.tgt_voc_size = tgt_voc_size
        self.start_token = nn.Parameter(torch.randn(1, 1, dim_model))

        # embedding layers
        self.tgt_embedding = nn.Embedding(tgt_voc_size, embedding_dim)
        
        self.mem_fc = nn.Linear(mem_input_size, dim_model)

        self.pe = PositionalEncoding(d_model=dim_model, dropout=dropout, max_len=max_len+1)

        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=num_heads, dim_feedforward=hidden_dim,
                                                               dropout=dropout, batch_first=True, norm_first=True)

        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_layers)

        self.fc_output = nn.Linear(in_features=dim_model, out_features=tgt_voc_size * num_q)


    def forward(self, mem, tgt, tgt_mask=None):
        mem, tgt = self.mem_fc(mem), self.tgt_embedding(tgt)
        tgt = torch.flatten(tgt, start_dim=2)

        start_tokens = self.start_token.repeat(tgt.shape[0], 1, 1)
        mem = torch.cat((start_tokens, mem), dim = 1)
        tgt = torch.cat((start_tokens, tgt), dim = 1)

        mem, tgt = self.pe(mem), self.pe(tgt)     
        
        transformer_output = self.transformer_decoder(tgt, mem, tgt_mask)
        logits = self.fc_output(transformer_output)

        # unflatten out
        logits = torch.unflatten(logits, -1, (self.num_q, self.tgt_voc_size))

        return logits