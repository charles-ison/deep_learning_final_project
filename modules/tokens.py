import torch

def get_tokens(residual_audio, tgt_audio, mert_processor, mert, encodec, resample_rate, device):
    # list_residual_audio = [row for row in residual_audio.numpy()]
    tgt_audio = tgt_audio.to(device)
    residual_audio = residual_audio.to(device)
    # mert.eval()
    encodec.eval()
    # forward passes
    with torch.no_grad():
        # audio_features = mert_processor(list_residual_audio, sampling_rate=resample_rate, do_normalize=False, return_tensors="pt")
        # mert_outputs = mert(**audio_features.to(device), output_hidden_states=True)
        tgt_codes = encodec(tgt_audio, return_encoded=False)
        mem_codes = encodec(residual_audio, return_encoded=False)

    tgt_tokens = tgt_codes[1].long().to(device)
    mem_tokens = mem_codes[1].long().to(device)

    # all_layer_hidden_states = torch.stack(mert_outputs.hidden_states)
    # semantic_tokens = all_layer_hidden_states[7] # picking layer 7 for now
    # acoustic_tokens = all_layer_hidden_states[-1]
    
    # mem_tokens = torch.cat((acoustic_tokens, semantic_tokens), 2).to(device)

    return mem_tokens, tgt_tokens