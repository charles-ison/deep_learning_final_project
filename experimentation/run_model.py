"""
Project: Conditional Musical Accompaniment Generation
File: run_model.py
Authors: John Zontos, Charles Ison, Shengxuan Wang
Date: June 4, 2023

Description:
This Python script, 'run_model.py', is part of a group project on Conditional Musical Accompaniment Generation.
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
from modules.audio_transformer_decoder import AudioTransformerDecoder
from audiolm_pytorch.encodec import EncodecWrapper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO: @shawn Finish DataLoader "data.py"
# NOTE: remember to check sample rate compatability.

# fake data
resample_rate = 24000
# TODO: @chase Use real audio here.
# TODO: @jc and chase Update to work with batch
residual_audio = torch.randn(2, 240000)
tgt_audio = torch.randn(2, 240000).to(device)
# stem_audio = residual_audio*3

# NOTE: mert_processor (Wav2Vec2FeatureExtractor) expects a batch to be a list of numpy arrays.
list_residual_audio = [row for row in residual_audio.numpy()]

# models
mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)
mert = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
encodec = EncodecWrapper().to(device)

# forward passes
audio_features = mert_processor(list_residual_audio, sampling_rate=resample_rate, do_normalize=False, return_tensors="pt")
mert_outputs = mert(**audio_features, output_hidden_states=True)
# TODO: @jc Check docs for start and end tokens.
emb, codes, _ = encodec(tgt_audio, return_encoded = True)

# tokens
# TODO: @jc Extract out seperate get_tokens.py
all_layer_hidden_states = torch.stack(mert_outputs.hidden_states)
# TODO: Imperically test different layers.
semantic_tokens = all_layer_hidden_states[7] # picking layer 7 for now
acoustic_tokens = all_layer_hidden_states[-1]
print("semantic_tokens.shape:", semantic_tokens.shape)
print("acoustic_tokens.shape", acoustic_tokens.shape)

# NOTE: @jc fix after adding batched input.
# NOTE: Could use embeddings instead of codes.
# tgt_tokens = codes.unsqueeze(dim=0).float().to(device)
tgt_tokens = codes.float().to(device)
print("tgt_tokens.shape:", tgt_tokens.shape)

# Define the encoder and decoder inputs, and the decoder output
encoder_input = torch.cat((acoustic_tokens, semantic_tokens), 2).to(device)
decoder_input = tgt_tokens[:, :-1]
decoder_output = tgt_tokens[:, 1:]

print("encoder_input.shape:", encoder_input.shape)
print("decoder_input.shape:", decoder_input.shape)
print("decoder_output.shape:", decoder_output.shape)

# Model training
enc_vocab_size = encoder_input.shape[-1]
dec_vocab_size = decoder_input.shape[-1]
hidden_dim = 256
dim_model = 512
num_layers = 4
num_heads = 4
dropout = 0.1
max_len = max(encoder_input.shape[1], decoder_input.shape[1])

# TODO: @chase see if pytorch.transformer handles autoregression.
# TODO: @jc investigate masking.
model = AudioTransformer(enc_vocab_size, dec_vocab_size, max_len, dim_model, hidden_dim, num_layers, num_heads, dropout).to(device)
# model = AudioTransformerDecoder(enc_vocab_size, dec_vocab_size, max_len, dim_model, hidden_dim, num_layers, num_heads, dropout).to(device)

# TODO: @jc investigate correct loss function. 
# NOTE: Look in MusicLM paper.
criterion = nn.MSELoss()

# Decoder only requires substantially lower learning rate, not sure if it indicates a bug
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop (example)
num_epochs = 10

for epoch in range(num_epochs):
    optimizer.zero_grad()
    predicted_codes = model(encoder_input, decoder_input)
    loss = criterion(predicted_codes, decoder_output)
    loss.backward(retain_graph=True)
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")


# TODO: Investigate inference time..