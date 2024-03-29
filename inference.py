from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
import torchaudio
import torch.nn as nn

import numpy as np
from scipy.spatial.distance import cdist
import librosa

import numpy as np
from scipy.spatial.distance import cdist
import librosa

from modules.audio_transformer import AudioTransformer
from modules.audio_transformer_decoder import AudioTransformerDecoder
from audiolm_pytorch.encodec import EncodecWrapper
from modules.data import TrackDataset
from modules.tokens import get_tokens
from modules.generate import generate_bass
from modules.positional_encoding import PositionalEncoding

# ---------- neptune ----------
NEPTUNE_SWITCH = 0
if NEPTUNE_SWITCH == 1:
    from neptune_init import runtime
    from neptune.utils import stringify_unsupported
# -----------------------------

print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------- params -----------
model_path = "model.pt"
test_data_dir = '/nfs/hpc/share/stemgen/slakh2100_wav_redux/test'
# test_data_dir = '/nfs/hpc/share/stemgen/sin_wave_test'
output_dir = "test/"

k=8
temp=0.99

num_examples = 20
window_size = 5
sample_rate = 24000
num_q = 8
# -----------------------------

# ---------- Dataset ----------
test_dataset = TrackDataset(test_data_dir)
test_dataset.set_window_size(window_size)
test_dataset.set_sample_rate(sample_rate)
print("INFO: Dataset loaded. Length:", len(test_dataset))
# -----------------------------

# ---------- pretrained models ----------
# mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)
# # TODO: Look into changing encoding length.
# mert = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to("cuda")
mert_processor = None
mert = None

encodec = EncodecWrapper().to(device)
codebook_size = 1024
num_q = encodec.num_quantizers
print("INFO: Encodec and Mert models loaded.")


# Our Model
model = torch.load(model_path)
model.eval()
print("INFO: Model created.")
# parallelize model
if torch.cuda.device_count() > 1:
    print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
    model = nn.DataParallel(model)

for sample_idx in range(num_examples):
    # get example
    residual_audio, target_audio = test_dataset[sample_idx]
    residual_audio = residual_audio.reshape(1, -1)
    target_audio = target_audio.reshape(1, -1)

    # tokens
    mem_tokens, tgt_tokens = get_tokens(residual_audio, target_audio, mert_processor, mert, encodec, sample_rate, device)
    # trimming extra encodec sample to match Mert.
    tgt = tgt_tokens[:, :-1]            # [B, timesteps, num_quantizers]
    mem = mem_tokens
    _, max_len, mem_emb_dim = mem.shape

    torchaudio.save(f"{output_dir}{sample_idx}_res.wav", residual_audio, sample_rate)
    print(f"INFO: {output_dir}{sample_idx}_res.wav saved.")
    torchaudio.save(f"{output_dir}{sample_idx}_tgt.wav", target_audio, sample_rate)
    print(f"INFO: {output_dir}{sample_idx}_tgt.wav saved.")

    pred_wav = generate_bass(model, encodec, mem, sample_idx, num_q, sample_rate, max_len, output_dir, device, k=k, temp=temp)

if NEPTUNE_SWITCH == 1:
    runtime["audio_files"].upload_files("*.wav")
    print(f"INFO: saved to neptune.")