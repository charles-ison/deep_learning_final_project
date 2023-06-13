import torch


def get_tokens(residual_audio, tgt_audio, mert_processor, mert, encodec, resample_rate, device):
    list_residual_audio = [row for row in residual_audio.numpy()]
    tgt_audio = tgt_audio.to(device)

    # forward passes
    audio_features = mert_processor(list_residual_audio, sampling_rate=resample_rate, do_normalize=False, return_tensors="pt")
    mert_outputs = mert(**audio_features, output_hidden_states=True)
    emb, codes, _ = encodec(tgt_audio, return_encoded = True)

    all_layer_hidden_states = torch.stack(mert_outputs.hidden_states)
    semantic_tokens = all_layer_hidden_states[7] # picking layer 7 for now
    acoustic_tokens = all_layer_hidden_states[-1]

    tgt_tokens = codes.long().to(device)

    return semantic_tokens, acoustic_tokens, tgt_tokens