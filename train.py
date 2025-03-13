import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tokenizers import Tokenizer
from pathlib import Path

from model import TransformerModel  
from dataset import QADataset 
from utils import get_num_params, print_config


def train_model(config):
    print_config(config)

    model = TransformerModel(config)
    print(f"Number of parameters in the model: {get_num_params(model):,}")

    if Path(config.model_filename).exists() and Path(config.optimizer_filename).exists():
        print("Loading model and optimizer state dicts...")
        model.load_state_dict(torch.load(config.model_filename, weights_only=True))
        model = model.to(config.device)
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        optimizer.load_state_dict(torch.load(config.optimizer_filename, weights_only=True))
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        model = model.to(config.device)

    tokenizer = Tokenizer.from_file(config.tokenizer_filename)
    dataset = QADataset(config, tokenizer)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.dataloader_num_workers, generator=torch.Generator().manual_seed(config.seed))

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in (pbar := tqdm(enumerate(train_loader), total=len(train_loader))):
            source, target, key_padding_mask = batch.values()
            source = source.to(config.device)
            target = target.to(config.device)
            key_padding_mask = key_padding_mask.to(config.device)

            out = model(source, padding_mask=key_padding_mask)
            loss = criterion(out.transpose(1, 2), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_description(f"[{epoch + 1:03} | {config.num_epochs:03}] Loss: {loss.item():.4f}")

            if (batch_idx + 1) % 500 == 0: # Save checkpoint every 500 batches
                torch.save(model.state_dict(), config.model_filename)
                torch.save(optimizer.state_dict(), config.optimizer_filename)

        mean_epoch_loss = total_loss / len(train_loader)
        print(f"\nMean Epoch Cross-Entropy Loss: {mean_epoch_loss}")

    torch.save(model.state_dict(), config.model_filename)
    torch.save(optimizer.state_dict(), config.optimizer_filename)

    return model


if __name__ == "__main__":
    from config import config

    train_model(config)
    
