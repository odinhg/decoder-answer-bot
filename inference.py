import torch

def top_p_sampling(last_token_logits, p=0.95, temperature=0.7):
    """
    Top-p sampling. Sample a random token from the smallest possible set of tokens whose cumulative probability exceeds the probability p. Temperature is applied before computing the probabilities with softmax (control smoothness of the distribution).
    """
    scaled_logits = last_token_logits / temperature                  
    probs = torch.softmax(scaled_logits, dim=-1).squeeze()
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                         
    if cumulative_probs[0] > p:                                               
        # Only one token contribute to the top p so return it
        return sorted_indices[0]
                                                         
    # More than one token contribute to the top p so sample one
    sorted_probs = sorted_probs[cumulative_probs <= p]
    sorted_indices = sorted_indices[cumulative_probs <= p]
    sampled_index = torch.multinomial(sorted_probs, num_samples=1)
                                         
    return sorted_indices[sampled_index]

def greedy_sampling(last_token_logits):
    """
    Greedy sampling. Select the token with the highest probability.
    """
    return torch.argmax(last_token_logits)

def sample_sequence(config, model, tokenizer, question_text):
    model.eval()
    with torch.no_grad():
        source = tokenizer.encode(question_text).ids
        source = tokenizer.token_to_id("[QST]") + source + tokenizer.token_to_id("[ANS]")
        source = torch.tensor(source).unsqueeze(0).to(config.device)

        generated_sequence = question_tokens
        answer = []
        for _ in range(config.max_len):
            last_token_logits = model(generated_sequence)
            last_token_logits = last_token_logits[0, -1, :]

            if config.sampling_strategy == "greedy":
                next_token = greedy_sampling(last_token_logits)
            elif config.sampling_strategy == "top-p":
                next_token = top_p_sampling(last_token_logits, p=config.top_p, temperature=config.temperature)
            else:
                raise ValueError("Invalid sampling strategy.")

            generated_sequence = torch.cat([generated_sequence, next_token.view(1, 1)], dim=1)
            answer.append(next_token.item())

            if next_token == tokenizer.token_to_id("[END]"):
                break

        answer_text = tokenizer.decode(answer, skip_special_tokens=True)
        return answer_text


if __name__ == "__main__":
    from config import config
    from tokenizers import Tokenizer
    from model import TransformerModel

    model = TransformerModel(config)
    model.load_state_dict(torch.load(config.model_filename, weights_only=True))
    model = model.to(config.device)

    tokenizer = Tokenizer.from_file(config.tokenizer_filename)

    question_text = "What is the capital of France?"
    answer_text = sample_sequence(config, model, tokenizer, question_text) 
    
    print(f"Question: {question_text}")
    print(f"Answer: {answer_text}")


