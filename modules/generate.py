import torch
import torchaudio
import torch.nn.functional as F



def temperature_scaling(logits, temperature):
    # leaving print statements for dependant stream sampling developement
    # NOTE: remove prints after developement

    # logits = torch.sum(logits, dim=-1)
    # print("logits:", logits.shape)
    scaled_logits = logits / temperature
    # print("scaled_logits:", scaled_logits)
    scaled_logits -= torch.max(scaled_logits, dim=-1, keepdim=True)[0]  # subtract the maximum for numerical stability
    scaled_probs = torch.exp(scaled_logits) / torch.sum(torch.exp(scaled_logits), dim=-1, keepdim=True)
    # print("scaled_probs:", scaled_probs)

    return scaled_probs


def top_k_sampling(probs, k):
    # leaving print statements for dependant stream sampling developement
    # NOTE: remove prints after developement

    # print("probs.shape:", probs.shape)
    batch_size, num_q, emb_dim = probs.shape
    topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)
    # print("topk_probs:", topk_probs)
    # print("topk_indices:", topk_indices)
    
    # Normalize probabilities to make them sum to 1
    normalized_probs = topk_probs / torch.sum(topk_probs, dim=-1, keepdim=True)
    normalized_probs = normalized_probs.reshape(-1, k)
    # print("normalized_probs:", normalized_probs)

    # Sampling from the top-k probabilities
    sample_idxs = torch.multinomial(normalized_probs, num_samples=1)
    sample_idxs = sample_idxs.reshape(batch_size, num_q, 1)
    # print("topk_indices.shape:", topk_indices.shape)
    # print("topk_indices:", topk_indices)
    # print("sample_idxs:", sample_idxs)
    sample = torch.gather(topk_indices, -1, sample_idxs)
    # print("sample", sample)
    # print("sample.shape", sample.shape)
    # exit()
    return sample.reshape(1, num_q)


def generate_bass(model, encodec, mem, sample_idx, num_q, sample_rate, device, k=1, temp=1.0):
    seq_length = mem.shape[1]

    # Perform inference
    with torch.no_grad():
        pred = torch.zeros(1, seq_length, num_q).long().to(device)

        # Generate sequence
        for i in range(seq_length):
            # Decode the next token
            decoder_output = model(mem, pred)

            # Get the mdoel output for the next time step.
            logits = decoder_output[:, i]

            # Apply temperature scaling
            scaled_probs = temperature_scaling(logits, temp)

            next_token = top_k_sampling(scaled_probs, k)
            
            # insert the token to the prediction
            pred[:, i] = next_token

    pred_wav = encodec.decode_from_codebook_indices(pred.to(device))
    pred_wav = pred_wav.reshape(1, -1).detach().cpu()
    print("pred_wav.shape:", pred_wav.shape)

    torchaudio.save(f"inference/{sample_idx}_out.wav", pred_wav, sample_rate)
    print(f"INFO: inference/{sample_idx}_out.wav saved.")