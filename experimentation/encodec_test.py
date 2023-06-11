import torch
from scipy.io.wavfile import write
from modules.data import TrackDataset
from torch.utils.data import DataLoader

from audiolm_pytorch.encodec import EncodecWrapper
encodec = EncodecWrapper()

resample_rate = 24000
# batch_size = 10
# ---------- Dataset ----------
train_data_dir = '/nfs/stak/users/zontosj/stemgen/slakh2100_wav_redux/validation'
train_dataset = TrackDataset(train_data_dir)
train_dataset.set_window_size(5)
train_dataset.set_sample_rate(resample_rate)

# train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
print("INFO: Dataset loaded. Length:", len(train_dataset))
# -----------------------------
res, tgt = train_dataset[0]
_, codes, _ = encodec(res, return_encoded = True) # (1, 10080) - 1 channel

print("res.shape:", res.shape)  # [batch_size, len(audio)]
# print("res:", res)    
print("codes.shape:", codes.shape)
# print("codes:", codes)        # [batch_size, timesteps, codebook_dim=8]

decoded = encodec.decode_from_codebook_indices(codes)
print(decoded.shape)
write("test.wav", resample_rate, decoded.detach().numpy())
