import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(), # Alternatively, use GELU activation
            nn.Linear(embed_size * 4, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask, padding_mask):
        attn_output, _ = self.attention(
            x, x, x, 
            attn_mask=attn_mask, 
            key_padding_mask=padding_mask, 
            need_weights=False,
            is_causal=True,
        )
        x = self.norm1(attn_output + x)
        ffn_out = self.ffn(x)
        out = self.norm2(ffn_out + x)
        return self.dropout(out)


class TransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, dropout, max_len):
        super().__init__()
        self.embed_size = embed_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = self.create_positional_encoding(max_len, embed_size)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([DecoderBlock(embed_size, num_heads, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, vocab_size)

        # Store reusable causal mask
        self.register_buffer("causal_mask", self.generate_causal_mask(max_len))

    def forward(self, x, padding_mask=None):
        batch_size, seq_len = x.shape

        # Use the precomputed causal mask (trim to match seq_len)
        attn_mask = self.causal_mask[:seq_len, :seq_len]

        # Inject positional encoding
        x = self.embedding(x) + self.positional_encoding[:seq_len, :].to(x.device)
        x = self.dropout(x)

        if padding_mask is None: # Assume no padding tokens
            padding_mask = torch.zeros(batch_size, seq_len).bool().to(x.device)

        for layer in self.layers:
            x = layer(x, attn_mask, padding_mask)

        return self.fc_out(x)

    def create_positional_encoding(self, max_len, embed_size):
        pos = torch.arange(max_len).unsqueeze(1)
        i = torch.arange(embed_size // 2).unsqueeze(0)
        angles = pos / torch.pow(10000, 2 * (i // 2) / embed_size)
        pos_encoding = torch.zeros(max_len, embed_size)
        pos_encoding[:, 0::2] = torch.sin(angles)
        pos_encoding[:, 1::2] = torch.cos(angles)
        return pos_encoding

    def generate_causal_mask(self, seq_len):
        """Generates an upper triangular mask to prevent attending to future tokens."""
        return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()


if __name__ == "__main__":
    from tokenizers import Tokenizer
    from datasets import load_dataset
    from utils import CodeDocstringDataset
    import config

    from torch.utils.data import DataLoader

    tokenizer = Tokenizer.from_file(config.tokenizer["tokenizer_file"])
    code_doc_dataset = CodeDocstringDataset(
        config.general["dataset"], tokenizer, config.model["max_len"]
    )

    model = TransformerDecoderOnly(
        vocab_size=config.tokenizer["vocab_size"],
        embed_size=config.model["embed_size"],
        num_heads=config.model["num_heads"],
        num_layers=config.model["num_layers"],
        dropout=config.model["dropout"],
        max_len=config.model["max_len"],
    )

    # print number of parameters in the model
    print(
        f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}"
    )

    dataloader = DataLoader(
        code_doc_dataset, batch_size=config.model["batch_size"], shuffle=False
    )

    batch = next(iter(dataloader))
    source_sequence, target_sequence, key_padding_mask = batch.values() 
    print("Source sequence shape:", source_sequence.shape)
    print("Target sequence shape:", target_sequence.shape)
    print("Key padding mask shape:", key_padding_mask.shape)

    out = model(source_sequence, padding_mask=key_padding_mask)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out.transpose(2, 1), target_sequence)
    print("Loss:", loss)
    loss.backward()
