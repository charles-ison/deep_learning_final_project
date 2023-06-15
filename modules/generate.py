import torch
import torchaudio


def generate_bass(model, encodec, mem, sample_idx, num_q, sample_rate, device):
    seq_length = mem.shape[1]

    # Perform inference
    with torch.no_grad():
        pred = torch.zeros(1, seq_length, num_q).long().to(device)

        # Generate sequence
        for i in range(seq_length):
            # Decode the next token
            decoder_output = model(mem, pred)

            # Get the most probable token
            next_token = torch.argmax(decoder_output, dim=-1)[:, i]
            
            # insert the token to the prediction
            pred[:, i] = next_token

    pred_wav = encodec.decode_from_codebook_indices(pred.to(device))
    pred_wav = pred_wav.reshape(1, -1).detach().cpu()
    print("pred_wav.shape:", pred_wav.shape)

    torchaudio.save(f"{sample_idx}_out.wav", pred_wav, sample_rate)
    print(f"INFO: {sample_idx}_out.wav saved.")