import torch

from audiolm_pytorch.encodec import EncodecWrapper
encodec = EncodecWrapper().cuda()

audio = torch.randn(10080).cuda()
recons = encodec(audio, return_recons_only = True) # (1, 10080) - 1 channel
print(recons)
