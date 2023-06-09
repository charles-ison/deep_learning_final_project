from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modules.audio_transformer import AudioTransformer
from modules.audio_transformer_decoder import AudioTransformerDecoder
from audiolm_pytorch.encodec import EncodecWrapper
from modules.data import TrackDataset
from modules.tokens import get_tokens

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resample_rate = 24000


# ---------- Dataset ----------
train_data_dir = 'data/mini/train'
train_dataset = TrackDataset(train_data_dir)
train_dataset.set_window_size(10)
train_dataset.set_sample_rate(24000)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# -----------------------------

# ---------- models ----------
mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)
mert = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
encodec = EncodecWrapper().to(device)
# -----------------------------

# get sizes
r, t = train_dataset[0]
get_tokens(r, t, mert_processor, mert, encodec, resample_rate, device)
enc_vocab_size = r.shape[-1]
dec_vocab_size = t.shape[-1]
max_len = max(r.shape[1], t.shape[1])

# ---------- params ----------
hidden_dim = 256
dim_model = 512
num_layers = 4
num_heads = 4
dropout = 0.1
num_epochs = 10
# -----------------------------

# Instantiate the model
model = AudioTransformer(enc_vocab_size, dec_vocab_size, max_len, dim_model, hidden_dim, num_layers, num_heads, dropout).to(device)
# model = AudioTransformerDecoder(enc_vocab_size, dec_vocab_size, max_len, dim_model, hidden_dim, num_layers, num_heads, dropout).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.MSELoss()


for epoch in range(num_epochs):
    for i, (residual_audio, tgt_audio) in enumerate(train_loader):
        # -------- get tokens ---------
        residual_audio = residual_audio.squeeze()
        tgt_audio = tgt_audio.squeeze()
        print("residual_audio:", residual_audio.shape)
        print("tgt_audio:", tgt_audio.shape)

        semantic_tokens, acoustic_tokens, tgt_tokens = get_tokens(residual_audio.squeeze(), tgt_audio.squeeze(), mert_processor, mert, encodec, resample_rate, device)
        # -----------------------------
        encoder_input = torch.cat((acoustic_tokens, semantic_tokens), 2).to(device)
        decoder_input = tgt_tokens[:, :-1]
        decoder_output = tgt_tokens[:, 1:]

        print("encoder_input.shape:", encoder_input.shape)
        print("decoder_input.shape:", decoder_input.shape)
        print("decoder_output.shape:", decoder_output.shape)

        optimizer.zero_grad()
        predicted_codes = model(encoder_input, decoder_input)
        loss = criterion(predicted_codes, decoder_output)
        loss.backward(retain_graph=True)
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
