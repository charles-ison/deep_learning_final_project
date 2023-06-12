from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.io.wavfile import write

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

batch_size=5
lr=1e-5
# -----------------------------

# ---------- Dataset ----------
train_data_dir = '/nfs/hpc/share/stemgen/slakh2100_wav_redux/validation'
train_dataset = TrackDataset(train_data_dir)
train_dataset.set_window_size(5)
train_dataset.set_sample_rate(resample_rate)

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
print("INFO: Dataset loaded. Length:", len(train_dataset))
# -----------------------------

# ---------- models ----------
mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)
# TODO: Look into changing encoding length.
mert = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
encodec = EncodecWrapper().to(device)
print("INFO: Pretrained models loaded.")

# NOTE: change filename
saved_model_filename = "DataParallel_saved_model.pt"
model = torch.load(saved_model_filename)
model.eval()
print("INFO: Model created:", model)

if torch.cuda.device_count() > 1:
    print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
    model = nn.DataParallel(model)
# -----------------------------

# get example
residual_audio, target_audio = next(iter(train_loader))
print("residual_audio.shape", residual_audio.shape)
print("target_audio.shape", target_audio.shape)
semantic_tokens, acoustic_tokens, tgt_tokens = get_tokens(residual_audio, target_audio, mert_processor, mert, encodec, resample_rate, device)

src = torch.cat((acoustic_tokens, semantic_tokens), 2).to(device)
print("src.shape:", src.shape)
print("tgt_tokens.shape:", tgt_tokens.shape)

seq_length = src.shape[1]
vocab_size = tgt_tokens.shape[-1]
tgt = torch.zeros(1, seq_length, vocab_size).cuda()

# Perform inference
for _ in range(seq_length):
    tgt = model(src, tgt)

output = tgt

print("output:", output)
print("output.shape:", output.shape)
print("tgt_tokens:", tgt_tokens)
print("tgt_tokens.shape:", tgt_tokens.shape)

# Uncomment after prediction codes
# decoded_audio = encodec.decode_from_codebook_indices(output)
# print("decoded_audio.shape:", decoded_audio.shape)
# write("decoded_audio.wav", resample_rate, decoded_audio.detach().numpy())