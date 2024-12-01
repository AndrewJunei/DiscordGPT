import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(model, device, context, max_new_tok):
    temperature = 0.8
    prev = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0)
    output = prev
    past = None

    for _ in range(max_new_tok):
        logits, past = model(prev, past=past)
        logits = logits[:, -1] / temperature 
        log_probs = F.softmax(logits, dim=-1)
        prev = torch.multinomial(log_probs, num_samples=1)
        output = torch.cat((output, prev), dim=1)
    return output
