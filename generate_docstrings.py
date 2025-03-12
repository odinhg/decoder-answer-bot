import torch
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer

import config
from transformer import TransformerDecoderOnly
from utils import format_code

def beam_search(
    model,  # Your transformer model
    input_seq,  # Initial input sequence (e.g., tokenized code)
    beam_width,  # Number of beams to keep at each step
    max_len,  # Maximum length of the generated sequence
    tokenizer,  # Tokenizer for encoding/decoding
    start_token_id,  # ID of the [START] token
    end_token_id,  # ID of the [END] token
    device="cpu",  # Device to run the model on
    temperature=1.0,  # Temperature for softmax (higher = more random)
):
    # Initialize beams with the input sequence and a score of 0
    beams = [(input_seq, 0.0)]  # Each beam is a (sequence, cumulative_log_prob) tuple

    for _ in range(max_len):
        new_beams = []

        for seq, score in beams:
            # Stop expanding this beam if it already ends with the [END] token
            if seq[-1] == end_token_id:
                new_beams.append((seq, score))
                continue

            # Get model predictions for the current sequence
            with torch.no_grad():
                logits = model(seq.unsqueeze(0).to(device), padding_mask=None)
                logits = logits[:, -1, :]  # Get logits for the last token
                logits = logits / temperature  # Apply temperature
                probs = F.softmax(logits, dim=-1)  # Convert to probabilities

            # Get the top-k tokens and their log probabilities
            topk_probs, topk_tokens = torch.topk(probs, beam_width, dim=-1)
            topk_probs = topk_probs.squeeze().cpu()
            topk_tokens = topk_tokens.squeeze().cpu()

            # Expand the current beam with the top-k tokens
            for i in range(beam_width):
                new_token = topk_tokens[i].unsqueeze(0)
                new_seq = torch.cat([seq, new_token], dim=-1)
                new_score = score + torch.log(topk_probs[i]).item()
                new_beams.append((new_seq, new_score))

        # Sort all new beams by their cumulative log probability
        new_beams.sort(key=lambda x: x[1], reverse=True)

        # Keep only the top-k beams
        beams = new_beams[:beam_width]

        # Check if all beams have ended with the [END] token
        if all(beam[0][-1] == end_token_id for beam in beams):
            break

    # Decode the best beam into text
    best_sequence = beams[0][0]  # Get the sequence with the highest score
    decoded_output = tokenizer.decode(best_sequence.tolist(), skip_special_tokens=True)
    #for beam in beams:
    #  print(tokenizer.decode(beam[0].tolist(), skip_special_tokens=True))
    return decoded_output


if __name__ == "__main__":
    tokenizer = Tokenizer.from_file(config.tokenizer["tokenizer_file"])
    model = TransformerDecoderOnly(
        vocab_size=config.tokenizer["vocab_size"],
        embed_size=config.model["embed_size"],
        num_heads=config.model["num_heads"],
        num_layers=config.model["num_layers"],
        dropout=config.model["dropout"],
        max_len=config.model["max_len"],
    )

    if Path(config.model["model_file"]).exists():
        model.load_state_dict(torch.load(config.model["model_file"], weights_only=True))
        model = model.to(config.model["device"])
    else:
        raise ValueError("Model file not found. Please train the model first.")

    input_code = """def max(a, b):
    if a > b:
        return a
    else:
        return b
    """
    input_code = format_code(input_code)  # Format the input code
    code_ids = tokenizer.encode(input_code).ids  # Tokenize the input code

    # Add special tokens (e.g., [START], [DOC], etc.)
    start_id = tokenizer.token_to_id("[START]")
    doc_id = tokenizer.token_to_id("[DOC]")
    input_seq = torch.tensor([start_id] + code_ids + [doc_id]).to("cpu")

    # Run beam search
    beam_width = 5
    max_len = config.model["max_len"] - len(input_seq)
    start_token_id = tokenizer.token_to_id("[START]")
    end_token_id = tokenizer.token_to_id("[END]")

    predicted_docstring = beam_search(
        model.to("cpu"),
        input_seq,
        beam_width,
        max_len,
        tokenizer,
        start_token_id,
        end_token_id,
        device="cpu",
        temperature=0.5,
    )

    print("Predicted docstring:")
    print(predicted_docstring)
