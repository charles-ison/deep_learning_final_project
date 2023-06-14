from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
import torchaudio
import torch.nn as nn
import json

from modules.audio_transformer import AudioTransformer
from modules.audio_transformer_decoder import AudioTransformerDecoder
from audiolm_pytorch.encodec import EncodecWrapper
from modules.data import TrackDataset
from modules.tokens import get_tokens

print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sample_rate = 24000

# ---------- params ----------
with open('model_config.json') as json_file:
    params = json.load(json_file)

hidden_dim = params["hidden_dim"]
embedding_dim = params["embedding_dim"]
num_layers = params["num_layers"]
num_heads = params["num_heads"]
dropout = params["dropout"]
num_epochs = params["num_epochs"]
sample_rate = params["sample_rate"]
batch_size = 1
lr = params["lr"]
# -----------------------------

# ---------- Dataset ----------
test_data_dir = '/nfs/hpc/share/stemgen/mini/train'
test_dataset = TrackDataset(test_data_dir)
test_dataset.set_window_size(5)
test_dataset.set_sample_rate(sample_rate)
print("INFO: Dataset loaded. Length:", len(test_dataset))
# -----------------------------

# ---------- models ----------
mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)
# TODO: Look into changing encoding length.
mert = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
encodec = EncodecWrapper().to(device)
codebook_size = 1024
num_q = encodec.num_quantizers
print("INFO: Encodec and Mert models loaded.")
# -----------------------------

# get example
sample_idx = 0
residual_audio, target_audio = test_dataset[sample_idx]
residual_audio = residual_audio.reshape(1, -1)
target_audio = target_audio.reshape(1, -1)

# tokens
semantic_tokens, acoustic_tokens, tgt_tokens = get_tokens(residual_audio, target_audio, mert_processor, mert, encodec, sample_rate, device)
mem = torch.cat((acoustic_tokens, semantic_tokens), 2).to(device)
_, max_len, mem_emb_dim = mem.shape
print("mem.shape:", mem.shape)

# load model
model = torch.load("model.pt")
model.eval()
print("INFO: Model created.")

if torch.cuda.device_count() > 1:
    print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
    model = nn.DataParallel(model)

torchaudio.save(f"residual_audio_{sample_idx}.wav", residual_audio, sample_rate)
print(f"INFO: residual_audio_{sample_idx}.wav saved.")
torchaudio.save(f"target_audio_{sample_idx}.wav", target_audio, sample_rate)
print(f"INFO: target_audio_{sample_idx}.wav saved.")

seq_length = mem.shape[1]

# Perform inference
with torch.no_grad():
    pred = torch.zeros(1, seq_length, num_q).long()

    # Generate sequence
    for i in range(seq_length):
        # Decode the next token
        decoder_output = model(mem, pred)

        # Get the most probable token
        next_token = torch.argmax(decoder_output, dim=-1)[:, i]
        
        # insert the token to the prediction
        pred[:, i] = next_token

pred_wav = encodec.decode_from_codebook_indices(pred.to(device))
pred_wav = pred_wav.reshape(1, -1).detach().cpu()
print("pred_wav.shape:", pred_wav.shape)

torchaudio.save(f"output_{sample_idx}.wav", pred_wav, sample_rate)
print(f"INFO: output_{sample_idx}.wav saved.")