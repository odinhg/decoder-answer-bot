import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tokenizers import Tokenizer
from pathlib import Path

import config
from transformer import TransformerDecoderOnly
from utils import CodeDocstringDataset

model = TransformerDecoderOnly(
    vocab_size=config.tokenizer["vocab_size"],
    embed_size=config.model["embed_size"],
    num_heads=config.model["num_heads"],
    num_layers=config.model["num_layers"],
    dropout=config.model["dropout"],
    max_len=config.model["max_len"],
)
print(
    f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}"
)

# Load model and optimizer state if it exists
if Path(config.model["model_file"]).exists() and Path(config.model["optimizer_file"]).exists():
    print("Loading model and optimizer states...")
    model.load_state_dict(torch.load(config.model["model_file"], weights_only=True))
    model = model.to(config.model["device"])
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)#config.model["lr"])
    optimizer.load_state_dict(torch.load(config.model["optimizer_file"], weights_only=True))
else:
    optimizer = optim.AdamW(model.parameters(), lr=config.model["lr"])
    model = model.to(config.model["device"])

criterion = nn.CrossEntropyLoss(ignore_index=-100)
tokenizer = Tokenizer.from_file(config.tokenizer["tokenizer_file"])
dataset = CodeDocstringDataset(
    config.general["dataset"],
    tokenizer,
    config.model["max_len"],
    fraction=config.model["train_fraction"],
)
train_loader = DataLoader(
    dataset,
    batch_size=config.model["batch_size"],
    shuffle=True,
    generator=torch.Generator().manual_seed(config.general["seed"]),
    num_workers=config.model["num_workers"],
)

# Training loop
for epoch in range(config.model["num_epochs"]):
    model.train()  # Set the model to training mode
    total_loss = 0

    for batch_idx, batch in (
        pbar := tqdm(enumerate(train_loader), total=len(train_loader))
    ):
        source_sequence, target_sequence, key_padding_mask = batch.values() 

        source_sequence = source_sequence.to(config.model["device"])
        target_sequence = target_sequence.to(config.model["device"])
        key_padding_mask = key_padding_mask.to(config.model["device"])

        out = model(source_sequence, padding_mask=key_padding_mask)

        loss = criterion(out.transpose(1, 2), target_sequence)

        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters

        total_loss += loss.item()
        pbar.set_description(f"[{epoch + 1:03}] Loss: {loss.item():.4f}")

        if (batch_idx + 1) % 500 == 0:
            # Save state for model and optimizer
            torch.save(model.state_dict(), config.model["model_file"])
            print(f"Saved model to {config.model['model_file']}")
            torch.save(optimizer.state_dict(), config.model["optimizer_file"])
            print(f"Saved optimizer state to {config.model['optimizer_file']}")

    # Print loss every epoch
    print(f"Mean Epoch Cross-Entropy Loss: {total_loss / len(train_loader)}")

