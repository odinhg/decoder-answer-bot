import torch
from types import SimpleNamespace

# Global parameters
general = {
    "seed": 0,
    "dataset": "odinhg/vault-py-filtered-short",
    "device": "cpu" if not torch.cuda.is_available() else "cuda",
}

# Tokenizer configuration
tokenizer = {
    "train_fraction": 1.0,
    "vocab_size": 10000,
    "min_frequency": 5,
    "unk_token": "[UNK]",
    "special_tokens": ["[QUESTION]", "[ANSWER]", "[EOS]", "[PAD]", "[UNK]"],
    "num_workers": 4,
    "filename": "temp/tokenizer.json",
}

# Model configuration
model = {
    "embed_size": 128,
    "num_heads": 4,
    "num_layers": 4,
    "dropout": 0.1,
}

# Training configuration
training = {
    "max_len": 128,
    "train_fraction": 1.0,
    "batch_size": 32,
    "num_workers": 4,
    "lr": 1e-4,
    "num_epochs": 10,
    "model_filename": "temp/model.pth",
    "optimizer_filename": "temp/optimizer.pth",
}

# Inference configuration
inference = {
    "mode": "greedy",  # "greedy" or "top-p"
    "p": 0.95,
    "temperature": 0.7,
}

config = SimpleNamespace(
    general=SimpleNamespace(**general),
    tokenizer=SimpleNamespace(**tokenizer),
    model=SimpleNamespace(**model),
    training=SimpleNamespace(**training),
    inference=SimpleNamespace(**inference),
)

print("Using configuration:")
for key, namespace in config.__dict__.items():
    print(f"{key}:")
    for subkey, value in namespace.__dict__.items():
        print(f"{subkey}: {value}")
