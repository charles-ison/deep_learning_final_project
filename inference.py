from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader

import tqdm as tq

from modules.audio_transformer import AudioTransformer
from modules.audio_transformer_decoder import AudioTransformerDecoder
from audiolm_pytorch.encodec import EncodecWrapper
from modules.data import TrackDataset
from modules.tokens import get_tokens

print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resample_rate = 24000

# ---------- params ----------
hidden_dim = 256
dim_model = 512
num_layers = 4
num_heads = 4
dropout = 0.1
num_epochs = 100 

batch_size=48
lr=1e-5
# -----------------------------

# ---------- Dataset ----------
train_data_dir = '/nfs/stak/users/zontosj/stemgen/slakh2100_wav_redux/validation'
train_dataset = TrackDataset(train_data_dir)
train_dataset.set_window_size(5)
train_dataset.set_sample_rate(resample_rate)

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
print("INFO: Dataset loaded. Length:", len(train_dataset))
# -----------------------------

# ---------- models ----------
# TODO: Put these models on multiple gpus?
mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)
# TODO: Look into checnging sequence length.
mert = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
encodec = EncodecWrapper().to(device)
print("INFO: Pretrained models loaded.")
# -----------------------------

# get sizes
res, tgt = train_dataset[0]
semantic_tokens, acoustic_tokens, tgt_tokens = get_tokens(res, res, mert_processor, mert, encodec, resample_rate, device)

encoder_input = torch.cat((acoustic_tokens, semantic_tokens), 2).to(device)
decoder_input = tgt_tokens[:, :-1]
decoder_output = tgt_tokens[:, 1:]


enc_vocab_size = encoder_input.shape[-1]
dec_vocab_size = decoder_input.shape[-1]
max_len = max(encoder_input.shape[1], decoder_input.shape[1])

num_time_steps = tgt_tokens.shape[1]

# Instantiate the model
# TODO: @Chase investigate masking.
saved_model_path = "jcmain/deep_learning_final_project/model.pt"
# model = AudioTransformer(enc_vocab_size, dec_vocab_size, max_len, dim_model, hidden_dim, num_layers, num_heads, dropout).to(device)
model = AudioTransformerDecoder(enc_vocab_size, dec_vocab_size, max_len, dim_model, hidden_dim, num_layers, num_heads, dropout).to(device)
model.eval()
# print("INFO: Model created:", model)

# Initialize the decoder input with zeros
# encoder_input = encoder_input
start_token = decoder_input[:, 0:1, :]
# decoder_input = torch.zeros(1, 1, dec_vocab_size).cuda()


print("encoder_input.shape", encoder_input.shape)
print("start_token.shape", start_token.shape)

# Perform inference
with torch.no_grad():
    predicted_codes = []
    # for _ in range(num_time_steps):
    output = model(encoder_input, start_token)
    predicted_codes = torch.cat((start_token, output), dim=1)

    # Update the decoder input with the current output
    predicted_codes = output

# Print the predicted codes
print("predicted_codes.shape", predicted_codes.shape)
print("predicted_codes", predicted_codes)
encodec = EncodecWrapper().to(device)
src_encodec_tokens = encodec(res, return_encoded = False)
src_encodec_tokens = src_encodec_tokens[1]
# audio = encodec.decode_from_codebook_indices(predicted_codes[0].unsqueeze(dim=0).long())
audio = encodec.decode_from_codebook_indices(src_encodec_tokens)
print(audio.shape)
audio = audio.view(1, -1).t().cpu()
torchaudio.save("test.wav", audio, resample_rate)

# sox test.wav -n spectrogram