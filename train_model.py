from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm as tq
import json

from modules.audio_transformer import AudioTransformer
from modules.audio_transformer_decoder import AudioTransformerDecoder
from audiolm_pytorch.encodec import EncodecWrapper
from modules.data import TrackDataset
from modules.tokens import get_tokens
from modules.positional_encoding import PositionalEncoding
from modules.generate import generate_bass

RUN_INFERENCE = 1

# ---------- neptune ----------
NEPTUNE_SWITCH = 1
if NEPTUNE_SWITCH == 1:
    from neptune_init import runtime
    from neptune.utils import stringify_unsupported
    log_every = 5          # log loss every 'log_every' steps
# -----------------------------

print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------- params ----------
with open('model_config.json') as json_file:
    params = json.load(json_file)

if NEPTUNE_SWITCH:
    runtime["params"] = stringify_unsupported(params)

hidden_dim = params["hidden_dim"]
embedding_dim = params["embedding_dim"]
num_layers = params["num_layers"]
num_heads = params["num_heads"]
dropout = params["dropout"]
num_epochs = params["num_epochs"]
sample_rate = params["sample_rate"]
batch_size = params["batch_size"]
lr = params["lr"]
weight_decay = params["weight_decay"]
num_q = params["num_quantizers"]
window_size = params["window_size"]
k = params["k"]
temp = params["temperature"]
# -----------------------------

# ---------- Datasets ----------
train_data_dir = '/nfs/hpc/share/stemgen/slakh2100_wav_redux/train'
train_dataset = TrackDataset(train_data_dir)
train_dataset.set_window_size(window_size)
train_dataset.set_sample_rate(sample_rate)

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
print("INFO: Train dataset loaded. Length:", len(train_dataset))

val_data_dir = '/nfs/hpc/share/stemgen/slakh2100_wav_redux/test'
val_dataset = TrackDataset(val_data_dir)
val_dataset.set_window_size(window_size)
val_dataset.set_sample_rate(sample_rate)

val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
print("INFO: Validation dataset loaded. Length:", len(val_dataset))

if RUN_INFERENCE == 1:
    inf_residual_audio, inf_tgt_audio = val_dataset[0]
    output_dir = "out/"
    inf_every = 5       # run inference every 'inf_every' epoch
# -----------------------------

# ---------- Models ----------
# TODO: Put these models on multiple gpus?
mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
# TODO: Look into checnging sequence length.
mert = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to("cuda")
encodec = EncodecWrapper(num_quantizers = num_q).to(device)
codebook_size = 1024
num_q = encodec.num_quantizers
print("INFO: Encodec and Mert models loaded.")
# -----------------------------

# get sizes
# TODO: @jc Add to trainer class? or make tokenizer class.
res, tgt = train_dataset[0]
res, tgt = res.unsqueeze(0), tgt.unsqueeze(0)
semantic_tokens, acoustic_tokens, tgt_tokens = get_tokens(res, tgt, mert_processor, mert, encodec, sample_rate, device)

src = torch.cat((acoustic_tokens, semantic_tokens), 2).to(device)
tgt = tgt_tokens[:, :-1]

src_emb_dim = src.shape[-1]
max_len = max(src.shape[1], tgt.shape[1])

# Instantiate the model
model = AudioTransformerDecoder(src_emb_dim, codebook_size, embedding_dim, num_q, hidden_dim, num_layers, num_heads, dropout).to(device)
print("INFO: Model created:", model)

if torch.cuda.device_count() > 1:
    print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
    model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

best_loss = None
tgt_mask = nn.Transformer.generate_square_subsequent_mask(max_len+1, device=device).unsqueeze(dim=0)


for epoch in range(num_epochs):
    train_loss = 0.0
    pbar = tq.tqdm(desc="Epoch {}".format(epoch+1), total=len(train_loader), unit="steps")
    for i, (residual_audio, tgt_audio) in enumerate(train_loader):
        # get tokens
        semantic_tokens, acoustic_tokens, tgt_tokens = get_tokens(residual_audio, tgt_audio, mert_processor, mert, encodec, sample_rate, device)
        mem = torch.cat((acoustic_tokens, semantic_tokens), 2).to(device)
        # trimming extra encodec sample to match Mert.
        tgt = tgt_tokens[:, :-1]            # [B, timesteps, num_quantizers]

        optimizer.zero_grad()

        predicted_codes = model(mem, tgt, max_len+1, tgt_mask=tgt_mask)  # [B, L, Q, V]
        loss = criterion(predicted_codes.permute(0, 3, 1, 2), tgt_tokens)
        train_loss += loss.item()
        
        # Log after every "log_every" steps
        if i % log_every == 0 and NEPTUNE_SWITCH == 1:
            runtime['train/loss'].log(loss)
        loss.backward()
        optimizer.step()

        pbar.update(1)
    pbar.close()
    epoch_loss = train_loss / len(train_loader)
    if NEPTUNE_SWITCH == 1:
        runtime['epoch/train/loss'].log(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # validation
    model.eval()

    val_loss = 0.0
    with torch.no_grad():
        for i, (residual_audio, tgt_audio) in enumerate(val_loader):
            # get tokens
            semantic_tokens, acoustic_tokens, tgt_tokens = get_tokens(residual_audio, tgt_audio, mert_processor, mert, encodec, sample_rate, device)
            mem = torch.cat((acoustic_tokens, semantic_tokens), 2).to(device)
            # trimming extra encodec sample to match Mert.
            tgt = tgt_tokens[:, :-1]            # [B, timesteps, num_quantizers]

            predicted_codes = model(mem, tgt, max_len+1, tgt_mask=tgt_mask)  # [B, L, Q, V]
            loss = criterion(predicted_codes.permute(0, 3, 1, 2), tgt_tokens)
            val_loss += loss.item()

            # Log after every "log_every" steps
            if i % log_every and NEPTUNE_SWITCH == 1:
                runtime["validation/loss"].log(loss)

        epoch_loss = val_loss / len(val_loader)
        if NEPTUNE_SWITCH == 1:
            runtime["epoch/validation/loss"].log(epoch_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {epoch_loss:.4f}")
    
    if not best_loss or epoch_loss < best_loss:
        # NOTE: look at alternative model saving strat
        print("Best loss achieved, saving model.")
        torch.save(model, "model.pt")
        if NEPTUNE_SWITCH == 1:
            runtime["model"].upload("model.pt")
            print(f"INFO: saved model to neptune.")
        best_loss = loss

    if epoch % inf_every == 0:
            semantic_tokens, acoustic_tokens, tgt_tokens = get_tokens(inf_residual_audio.unsqueeze(0), inf_tgt_audio.unsqueeze(0), mert_processor, mert, encodec, sample_rate, device)
            mem = torch.cat((acoustic_tokens, semantic_tokens), 2).to(device)
            tgt = tgt_tokens[:, :-1] 

            generate_bass(model, encodec, mem[0].unsqueeze(0), epoch, num_q, sample_rate, max_len, output_dir, device, k=k, temp=temp)

            if NEPTUNE_SWITCH == 1:
                runtime["audio_files"].upload_files([f"out/{epoch}_out.wav"])
                print(f"INFO: saved audio to neptune.")

