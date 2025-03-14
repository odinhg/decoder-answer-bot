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

def sample_sequence(model, tokenizer, question_text, strategy, max_len, device, p=0.95, temperature=0.7):
    model.eval()
    with torch.no_grad():
        source = tokenizer.encode(question_text).ids
        source = torch.tensor(source).unsqueeze(0).to(device)

        question_tokens = tokenizer.encode("[QST]").ids + tokenizer.encode(question_text).ids + tokenizer.encode("[ANS]").ids
        question_tokens = torch.tensor(question_tokens).unsqueeze(0).to(device)

        generated_sequence = question_tokens
        answer = []
        for _ in range(max_len):
            last_token_logits = model(generated_sequence)
            last_token_logits = last_token_logits[0, -1, :]

            if strategy == "greedy":
                next_token = greedy_sampling(last_token_logits)
            elif strategy == "top-p":
                next_token = top_p_sampling(last_token_logits, p=p, temperature=temperature)
            else:
                raise ValueError("Invalid sampling strategy.")

            generated_sequence = torch.cat([generated_sequence, next_token.view(1, 1)], dim=1)
            answer.append(next_token.item())

            if next_token == tokenizer.token_to_id("[END]") or generated_sequence.size(1) >= max_len:
                break

        answer_text = tokenizer.decode(answer)
        return answer_text


if __name__ == "__main__":
    from config import config
    from tokenizers import Tokenizer
    from model import TransformerModel

    model = TransformerModel(config)
    model = model.to(config.device)
    model = torch.compile(model)
    model.load_state_dict(torch.load(config.model_filename, weights_only=True, map_location=config.device))

    tokenizer = Tokenizer.from_file(config.tokenizer_filename)

    question_text = "what is the largest dog breed?"

    print("Greedy sampling:")
    answer_text = sample_sequence(model, tokenizer, question_text, "greedy", 100, config.device)
    print(f"Question: {question_text}")
    print(f"Answer: {answer_text}")

    print("Top-p sampling (p=0.95, temperature=0.7):")
    answer_text = sample_sequence(model, tokenizer, question_text, "top-p", 100, config.device, p=0.95, temperature=0.7)
    print(f"Question: {question_text}")
    print(f"Answer: {answer_text}")


