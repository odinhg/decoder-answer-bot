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

def sample_sequence(input_sequence, model, strategy, max_len, device, end_id, p=0.95, temperature=0.7):
    model.eval()
    with torch.no_grad():
        input_sequence = input_sequence.unsqueeze(0).to(device) # Add batch dimension and move to device
        answer = []
        for _ in range(max_len):
            last_token_logits = model(input_sequence)
            last_token_logits = last_token_logits[0, -1, :]

            if strategy == "greedy":
                next_token = greedy_sampling(last_token_logits)
            elif strategy == "top-p":
                next_token = top_p_sampling(last_token_logits, p=p, temperature=temperature)
            else:
                raise ValueError("Invalid sampling strategy.")

            input_sequence = torch.cat([input_sequence, next_token.view(1, 1)], dim=1)
            answer.append(next_token.item())

            if next_token == end_id or input_sequence.size(1) >= max_len:
                break

        return answer

def tokenize_input(tokenizer, text, sep_id):
    """
    Tokenize input text and add special tokens.
    """
    tokens = tokenizer.encode(text).ids
    tokens = tokens + [sep_id]
    return torch.tensor(tokens)

def decode_output(tokenizer, tokens):
    """
    Decode output tokens.
    """
    return tokenizer.decode(tokens)

if __name__ == "__main__":
    from config import config
    from tokenizers import Tokenizer
    from model import TransformerModel

    model = TransformerModel(config)
    model = model.to(config.device)
    model = torch.compile(model)
    model.load_state_dict(torch.load(config.model_filename, weights_only=True, map_location=config.device))

    tokenizer = Tokenizer.from_file(config.tokenizer_filename)

    sep_id = tokenizer.token_to_id(config.sep_token)
    end_id = tokenizer.token_to_id(config.end_token)

    question_text = "what is the largest dog breed?"

    input_sequence = tokenize_input(tokenizer, question_text, sep_id) 

    print("Greedy sampling:")
    answer = sample_sequence(input_sequence, model, "greedy", 100, config.device, end_id)
    answer_text = decode_output(tokenizer, answer)
    print(f"Question: {question_text}")
    print(f"Answer: {answer_text}")

    print("Top-p sampling (p=0.95, temperature=0.7):")
    answer = sample_sequence(input_sequence, model, "top-p", 100, config.device, end_id, p=0.95, temperature=0.7)
    answer_text = decode_output(tokenizer, answer)
    print(f"Question: {question_text}")
    print(f"Answer: {answer_text}")


