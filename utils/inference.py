from torch.nn import functional as F
import torch

def beam_search_step(model, idx, max_new_tokens, k, device, context_length=256):

    sequences = [(idx.to("cpu"), 0)]  # Ensure the initial sequences are on the correct device
    for _ in range(max_new_tokens):
        all_candidates = []
        for seq, score in sequences:
            seq = seq.to(device) 
            seq = seq[:, -context_length:]
            logits = model.forward(seq)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            top_probs, top_idx = torch.topk(probs, k)
            for i in range(k):
                candidate_seq = torch.cat((seq, top_idx[:, i].unsqueeze(1)), dim=1)
                candidate_score = score - torch.log(top_probs[:, i]).item()  # Convert to Python float
                all_candidates.append((candidate_seq, candidate_score))
        if device == "cuda":
            torch.cuda.empty_cache()
        sequences = sorted(all_candidates, key=lambda tup: tup[1])[:k]
    return sequences


    
def predict(model, tokenizer, query, device, max_new_tokens = 100, k=5):
    input_test= torch.tensor(tokenizer.encode(query), dtype=torch.long).to(device)
    input_test = input_test.unsqueeze(0)
    output = beam_search_step(model, input_test, max_new_tokens, k, device)
    sequence = output[0][0]
    sequence_list = sequence[0].tolist()
    return tokenizer.decode(sequence_list)