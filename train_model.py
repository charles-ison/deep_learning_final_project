from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import tqdm as tq

from modules.audio_transformer import AudioTransformer
from modules.audio_transformer_decoder import AudioTransformerDecoder
from audiolm_pytorch.encodec import EncodecWrapper
from modules.data import TrackDataset
from modules.tokens import get_tokens

print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NEPTUNE_SWITCH = 1
# ---------- neptune ----------
if NEPTUNE_SWITCH == 1:
    from neptune_init import runtime

resample_rate = 24000

# ---------- params ----------
hidden_dim = 256
dim_model = 512
num_layers = 4
num_heads = 4
dropout = 0.1
num_epochs = 1

batch_size=48
lr=1e-5
# -----------------------------

# ---------- Dataset ----------
train_data_dir = '/nfs/hpc/share/stemgen/slakh2100_wav_redux/test'
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
# TODO: @jc Add to trainer class? or make tokenizer class.
res, tgt = train_dataset[0]
semantic_tokens, acoustic_tokens, tgt_tokens = get_tokens(res, tgt, mert_processor, mert, encodec, resample_rate, device)

src = torch.cat((acoustic_tokens, semantic_tokens), 2).to(device)
tgt = tgt_tokens[:, :-1]
decoder_output = tgt_tokens[:, 1:]

enc_vocab_size = src.shape[-1]
dec_vocab_size = tgt.shape[-1]
max_len = max(src.shape[1], tgt.shape[1])

# Instantiate the model
# model = AudioTransformer(enc_vocab_size, dec_vocab_size, max_len, dim_model, hidden_dim, num_layers, num_heads, dropout).to(device)
model = AudioTransformerDecoder(enc_vocab_size, dec_vocab_size, max_len, dim_model, hidden_dim, num_layers, num_heads, dropout).to(device)
print("INFO: Model created:", model)

if torch.cuda.device_count() > 1:
    print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
    model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr)

# TODO: @jc investigate correct loss function. (try CrossEntropyLoss and prediction codebook logits)
criterion = nn.MSELoss()

best_loss = None

for epoch in range(num_epochs):
    pbar = tq.tqdm(desc="Epoch {}".format(epoch+1), total=len(train_loader), unit="steps")
    for i, (residual_audio, tgt_audio) in enumerate(train_loader):
        # -------- get tokens ---------
        # Ensure residual_audio and tgt_audio have batch dimension first
        residual_audio = residual_audio if residual_audio.dim() == 2 else residual_audio.squeeze()
        tgt_audio = tgt_audio if tgt_audio.dim() == 2 else tgt_audio.squeeze()

        semantic_tokens, acoustic_tokens, tgt_tokens = get_tokens(residual_audio, tgt_audio, mert_processor, mert, encodec, resample_rate, device)
        # -----------------------------
        src = torch.cat((acoustic_tokens, semantic_tokens), 2).to(device)
        tgt = tgt_tokens[:, :-1]
        decoder_output = tgt_tokens[:, 1:]

        optimizer.zero_grad()
        predicted_codes = model(src, tgt)
        loss = criterion(predicted_codes, decoder_output)
        if NEPTUNE_SWITCH == 1:
            runtime['train/loss'].log(loss)
        loss.backward(retain_graph=True)
        optimizer.step()

        pbar.update(1)
    pbar.close()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    if not best_loss or loss < best_loss:
        # NOTE: look at alternative model saving strat
        torch.save(model, model.__class__.__name__ + "_saved_model.pt")
        best_loss = loss
