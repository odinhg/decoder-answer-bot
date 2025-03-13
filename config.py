import torch
from types import SimpleNamespace

config = {
    "seed": 0,
    "dataset": "odinhg/gooaq-subset",
    "split": "train",
    "device": "cpu" if not torch.cuda.is_available() else "cuda",
    "tokenizer_train_fraction": 1.0,
    "vocab_size": 15000,
    "min_frequency": 5,
    "unk_token": "[UNK]",
    "special_tokens": ["[QST]", "[ANS]", "[END]", "[PAD]", "[UNK]"],
    "tokenizer_filename": "temp/tokenizer.json",
    "embed_size": 128,
    "num_heads": 4,
    "num_layers": 4,
    "dropout_p": 0.1,
    "max_len": 128,
    "model_train_fraction": 1.0,
    "batch_size": 64,
    "dataloader_num_workers": 2,
    "lr": 1e-4,
    "num_epochs": 5,
    "model_filename": "temp/model.pth",
    "optimizer_filename": "temp/optimizer.pth",
    "inference_mode": "greedy",  # "greedy" or "top-p"
    "inference_p": 0.95,
    "inference_temperature": 0.7,
}

config = SimpleNamespace(**config)

