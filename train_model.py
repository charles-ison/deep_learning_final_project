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

# ---------- neptune ----------
NEPTUNE_SWITCH = 0
if NEPTUNE_SWITCH == 1:
    from neptune_init import runtime
# -----------------------------

print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------- params ----------
hidden_dim = 256
dim_model = 512
embedding_dim = 10
num_layers = 4
num_heads = 4
dropout = 0.1
num_epochs = 1
resample_rate = 24000
batch_size=4            # must be divisable by num_gpu
lr=1e-5
# -----------------------------

# ---------- Dataset ----------
# NOTE: developing on babyset need to change for training.
train_data_dir = '/nfs/hpc/share/stemgen/babyslakh_16k'
train_dataset = TrackDataset(train_data_dir)
train_dataset.set_window_size(5)
train_dataset.set_sample_rate(resample_rate)

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
print("INFO: Dataset loaded. Length:", len(train_dataset))
# -----------------------------

# ---------- models ----------
# TODO: Put these models on multiple gpus?
mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
# TODO: Look into checnging sequence length.
mert = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
encodec = EncodecWrapper().to(device)
codebook_size = 1024
num_q = encodec.num_quantizers
print("INFO: Pretrained models loaded.")
# -----------------------------

# get sizes
# TODO: @jc Add to trainer class? or make tokenizer class.
res, tgt = train_dataset[0]
res, tgt = res.unsqueeze(0), tgt.unsqueeze(0)
semantic_tokens, acoustic_tokens, tgt_tokens = get_tokens(res, tgt, mert_processor, mert, encodec, resample_rate, device)

src = torch.cat((acoustic_tokens, semantic_tokens), 2).to(device)
tgt = tgt_tokens[:, :-1]
# decoder_output = tgt_tokens[:, 1:]

src_emb_dim = src.shape[-1]
max_len = max(src.shape[1], tgt.shape[1])

# Instantiate the model
#model = AudioTransformer(enc_vocab_size, dec_vocab_size, max_len, dim_model, hidden_dim, num_layers, num_heads, dropout).to(device)
model = AudioTransformerDecoder(src_emb_dim, codebook_size, max_len, embedding_dim, num_q, hidden_dim, num_layers, num_heads, dropout).to(device)
print("INFO: Model created:", model)

if torch.cuda.device_count() > 1:
    print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
    model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr)
criterion = nn.CrossEntropyLoss()

best_loss = None
# add back max_len + 1 when start token is fixed.
tgt_mask = nn.Transformer.generate_square_subsequent_mask(max_len, device=device)
tgt_mask = tgt_mask.unsqueeze(dim=0)
tgt_mask = tgt_mask.repeat(num_heads * batch_size, 1, 1)
print("tgt_mask.shape:", tgt_mask.shape)

for epoch in range(num_epochs):
    pbar = tq.tqdm(desc="Epoch {}".format(epoch+1), total=len(train_loader), unit="steps")
    for i, (residual_audio, tgt_audio) in enumerate(train_loader):
        # -------- get tokens ---------
        semantic_tokens, acoustic_tokens, tgt_tokens = get_tokens(residual_audio, tgt_audio, mert_processor, mert, encodec, resample_rate, device)
        mem = torch.cat((acoustic_tokens, semantic_tokens), 2).to(device)
        tgt = tgt_tokens[:, :-1]            # [B, timesteps-1, num_quantizers=8]
        gt_tgt = tgt_tokens[:, 1:]          # [B, timesteps-1, num_quantizers=8]

        optimizer.zero_grad()
        # NOTE: removed mask for testing put back later.
        print("input mem.shape:", mem.shape)
        print("input tgt.shape:", tgt.shape)
        predicted_codes = model(mem, tgt, tgt_mask=tgt_mask)  # should be [B, L, Q, V]
        loss = criterion(predicted_codes, gt_tgt)
        if NEPTUNE_SWITCH == 1:
            runtime['train/loss'].log(loss)
        loss.backward(retain_graph=True)
        optimizer.step()

        pbar.update(1)
    pbar.close()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    if not best_loss or loss < best_loss:
        # NOTE: look at alternative model saving strat
        torch.save(model, "model.pt")
        best_loss = loss
