"""
Project: Conditional Musical Accompaniment Generation
File: stemgen.py
Authors: John Zontos, Charles Ison, Shengxuan Wang
Date: June 4, 2023

Description:
This Python script, 'stemgen.py', is part of a group project on Conditional Musical Accompaniment Generation.
The goal of this project is to develop a system that generates a bass track given some residual audio.

Authors:
- John Zontos: Responsible for Mert, Encodec, and AudioTransformer models
- Charles Ison: Acoustic Transformer
- Shengxuan Wang: Data pre-processing and loading

This script serves as the main entry point for the Conditional Musical Accompaniment Generation system.
"""
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
import torch.nn as nn

from modules.audio_transformer import AudioTransformer
from audiolm_pytorch.encodec import EncodecWrapper

# fake data
resample_rate = 24000
residual_audio = torch.randn(240000).cuda()
stem_audio = torch.randn(240000).cuda()
# stem_audio = residual_audio*3

# models
mert = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True) # required for mert
encodec = EncodecWrapper().cuda()

# forward passes
audio_features = processor(residual_audio, sampling_rate=resample_rate, do_normalize=False, return_tensors="pt")
print(audio_features)
outputs = mert(**audio_features, output_hidden_states=True)
emb, codes, _ = encodec(stem_audio, return_encoded = True)

# tokens
all_layer_hidden_states = torch.stack(outputs.hidden_states)
semantic_tokens = all_layer_hidden_states[7] # picking layer 7 for now
acoustic_tokens = outputs.last_hidden_state


print(semantic_tokens.shape)
print(acoustic_tokens.shape)
codes = codes.unsqueeze(dim=0).float().cuda()
print(codes.shape)

# Define the encoder and decoder inputs, and the decoder output
encoder_input = torch.cat((acoustic_tokens, semantic_tokens), 1).cuda()
decoder_input = codes[:, :-1]
decoder_output = codes[:, 1:]

print(encoder_input.shape)
print(decoder_input.shape)
print(decoder_output.shape)

# Model training
enc_vocab_size = encoder_input.shape[-1]
dec_vocab_size = decoder_input.shape[-1]
hidden_dim = 256
dim_model = 512
num_layers = 4
num_heads = 4
dropout = 0.1

model = AudioTransformer(enc_vocab_size, dec_vocab_size, dim_model, hidden_dim, num_layers, num_heads, dropout).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Training loop (example)
num_epochs = 10

for epoch in range(num_epochs):
    optimizer.zero_grad()
    predicted_codes = model(encoder_input, decoder_input)
    loss = criterion(predicted_codes, decoder_output)
    loss.backward(retain_graph=True)
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")