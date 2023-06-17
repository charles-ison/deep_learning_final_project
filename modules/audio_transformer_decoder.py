import torch
import math
import torch.nn as nn

class AudioTransformerDecoder(nn.Module):
    def __init__(self, mem_input_size, tgt_voc_size, embedding_dim, num_q, hidden_dim, num_layers, num_heads, dropout):
        super(AudioTransformerDecoder, self).__init__()
        dim_model = embedding_dim * num_q
        self.dim_model = dim_model
        self.num_q = num_q
        self.num_heads = num_heads
        self.tgt_voc_size = tgt_voc_size
        self.start_token = nn.Parameter(torch.randn(1, 1, dim_model))
        self.dropout = nn.Dropout(p=dropout)

        # embedding layers
        self.tgt_embedding = nn.Embedding(tgt_voc_size, embedding_dim)
        
        self.mem_embedding = nn.Embedding(tgt_voc_size, embedding_dim)
        # self.mem_fc = nn.Linear(mem_input_size, dim_model)

        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=num_heads, dim_feedforward=hidden_dim,
                                                               dropout=dropout, batch_first=True, norm_first=True)

        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_layers)

        self.fc_output = nn.Linear(in_features=dim_model, out_features=tgt_voc_size * num_q)

    # Adapted from here: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    # TODO: If there is time, use relative positional embeddings instead
    #   See: https://arxiv.org/pdf/1803.02155.pdf and https://arxiv.org/pdf/1809.04281.pdf
    def positional_encoding(self, max_len):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim_model, 2) * (-math.log(10000.0) / self.dim_model))
        pe = torch.zeros(1, max_len, self.dim_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, mem, tgt, max_len, tgt_mask=None):
        if tgt_mask != None:
            tgt_mask = tgt_mask.repeat(self.num_heads * tgt.shape[0], 1, 1)

        mem, tgt = self.mem_embedding(mem), self.tgt_embedding(tgt)

        tgt = tgt.permute(0, 1, 3, 2)
        tgt = torch.flatten(tgt, start_dim=2)

        mem = mem.permute(0, 1, 3, 2)
        mem = torch.flatten(mem, start_dim=2)

        start_tokens = self.start_token.repeat(tgt.shape[0], 1, 1)
        mem = torch.cat((start_tokens, mem), dim = 1)
        tgt = torch.cat((start_tokens, tgt), dim = 1)

        pe = self.positional_encoding(max_len)
        mem = mem + pe.to(mem.device)
        tgt = tgt + pe.to(tgt.device)

        transformer_output = self.transformer_decoder(tgt, mem, tgt_mask)
        logits = self.fc_output(transformer_output)

        # unflatten out
        logits = torch.unflatten(logits, -1, (self.num_q, self.tgt_voc_size))

        return logits[:, :-1]