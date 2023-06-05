import torch

from audiolm_pytorch.encodec import EncodecWrapper
encodec = EncodecWrapper().cuda()

audio = torch.randn(10080).cuda().unsqueeze(dim=0)
emb, codes, _ = encodec(audio, return_encoded = True) # (1, 10080) - 1 channel

print(audio.shape)  # [batch_size, len(audio)]
print(audio)
print(emb.shape)    # [batch_size, timesteps, emb_dim]
print(emb)    
print(codes.shape)
print(codes)        # [batch_size, timesteps, codebook_dim]

# embs = encodec.decode_from_codebook_indices(codes)
# print(emb.shape)
# print(embs)
# decoded = encodec.decode(emb)
# print(decoded.shape)
# print(decoded)