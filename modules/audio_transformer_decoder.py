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

        self.pe = PositionalEncoding(d_model=dim_model, dropout=dropout, max_len=max_len)

        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=num_heads, dim_feedforward=hidden_dim,
                                                               dropout=dropout, batch_first=True, norm_first=True)

        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_layers)

        self.fc_output = nn.Linear(in_features=dim_model, out_features=tgt_voc_size * num_q)

        self.log_softmax = nn.LogSoftmax(dim=-1)


    def forward(self, mem, tgt, tgt_mask=None):
        # start_tokens = self.start_token.repeat(tgt.shape[0], 1, 1)
        # mem = torch.cat((start_tokens, mem), dim = 1)
        # tgt = torch.cat((start_tokens, tgt), dim = 1)

        # mem.shape == [Batchsize, samples, mem_input_size]
        # tgt.shape == [Batchsize, samples, num_q]

        mem, tgt = self.mem_fc(mem), self.tgt_embedding(tgt.int())
        print("mem.shape:", mem.shape)
        print("tgt.shape:", tgt.shape)
            # mem = [Batchsize, samples, dim_model]
            # tgt = [Batchsize, samples, num_q, embedding_dim]
        tgt = torch.flatten(tgt, start_dim=2)
        print("tgt.shape:", tgt.shape)
            # tgt = [Batchsize, samples, dim_model]
        mem, tgt = self.pe(mem), self.pe(tgt)
            # tgt = [Batchsize, samples, dim_model]        
        
        transformer_output = self.transformer_decoder(tgt, mem, tgt_mask)
        output = self.fc_output(transformer_output)

        # unflatten out
        output = torch.unflatten(output, -1, (self.num_q, self.tgt_voc_size))

        # Removing end token
        output = output[:, 0:self.max_len, :]

        return self.log_softmax(output)