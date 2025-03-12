import torch
from pathlib import Path
from tokenizers import Tokenizer

import config
from transformer import TransformerDecoderOnly
from utils import format_code


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
    input_code = format_code(input_code)
    print("Formatted input code:")
    print(input_code)

    code_ids = tokenizer.encode(input_code).ids
    doc_id = tokenizer.token_to_id("[DOC]")
    start_id = tokenizer.token_to_id("[START]")
    end_id = tokenizer.token_to_id("[END]")

    seq = [start_id] + code_ids + [doc_id] + tokenizer.encode("Compute").ids
    seq = torch.tensor(seq).unsqueeze(0).to(config.model["device"])

    output = []

    while seq.size(1) < config.model["max_len"]:
        logits = model(seq, padding_mask=None)
        last_token_logits = logits[:, -1, :]
        probs = torch.softmax(last_token_logits, dim=-1)
        next_token = torch.argmax(
            probs, dim=-1
        )  # Greedy approach, can be changed to sampling
        seq = torch.cat([seq, next_token.unsqueeze(0)], dim=1)
        output.append(next_token)
        #if next_token == end_id:
        #    break

    predicted_docstring = tokenizer.decode(output, skip_special_tokens=False)
    print("Predicted docstring:")
    print(predicted_docstring)

