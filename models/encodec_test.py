import torch

from audiolm_pytorch.encodec import EncodecWrapper
encodec = EncodecWrapper().cuda()

audio = torch.randn(10080).cuda()
_, codes, _ = encodec(audio, return_encoded = False) # (1, 10080) - 1 channel

print(audio.shape)
# print(emb.shape)
print(codes.shape)