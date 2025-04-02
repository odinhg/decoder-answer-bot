import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # Using a fixed hidden layer size of 4 * embed_size
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask, padding_mask):
        if padding_mask is None: # Create "empty" padding mask if not provided
            padding_mask = torch.zeros(x.shape[:2]).bool().to(x.device)

        x_norm = self.norm1(x)

        attn_output, _ = self.attention(
            x_norm, x_norm, x_norm, 
            attn_mask=attn_mask, 
            key_padding_mask=padding_mask, 
            need_weights=False,
            is_causal=True,
        )

        attn_output = self.dropout1(attn_output)
        x = attn_output + x
        x = self.norm2(x)
        mlp_out = self.ffn(x)
        out = x + self.dropout2(mlp_out)
        return out


class PositionalEncoding(nn.Module):
    """
    Positional encoding module: adds positional information to the input embeddings.
    """
    def __init__(self, embed_size, max_len):
        super().__init__()
        positional_encoding = torch.zeros(max_len, embed_size)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = 1 / (10000 ** (torch.arange(0, embed_size, 2).float() / embed_size))  # Direct computation
        positional_encoding[:, 0::2] = torch.sin(pos * div_term)
        positional_encoding[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("positional_encoding", positional_encoding.unsqueeze(0))  # Store as buffer

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1), :].to(x.device)


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_size = config.embed_size
        self.num_layers = config.num_layers 
        self.vocab_size = config.vocab_size
        self.max_len = config.max_len
        self.dropout_p = config.dropout_p
        self.num_heads = config.num_heads
        self.device = config.device

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.pos_encoder = PositionalEncoding(self.embed_size, self.max_len)

        self.layers = nn.ModuleList([DecoderBlock(self.embed_size, self.num_heads, self.dropout_p) for _ in range(self.num_layers)])
        self.fc_out = nn.Linear(self.embed_size, self.vocab_size)

        # Precompute the causal mask and positional encoding
        self.register_buffer("causal_mask", self.generate_causal_mask(self.max_len))

    def forward(self, x, padding_mask=None):
        batch_size, seq_len = x.shape

        # Use the precomputed causal mask (trim to match seq_len)
        attn_mask = self.causal_mask[:seq_len, :seq_len]

        # Embed and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attn_mask, padding_mask)

        return self.fc_out(x)

    def generate_causal_mask(self, seq_len):
        """Generates an upper triangular mask to prevent attending to future tokens."""
        return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()


if __name__ == "__main__":
    from tokenizers import Tokenizer
    from torch.nn.functional import cross_entropy

    from config import config
    from utils import get_num_params
    from dataset import QADataset

    model = TransformerModel(config)
    print(f"Number of parameters in the model: {get_num_params(model):,}")

    # Simple forward pass for sanity checking
    tokenizer = Tokenizer.from_file(config.tokenizer_filename)
    dataset = QADataset(config, tokenizer)
    source = dataset[0]["source_sequence"].unsqueeze(0)
    target = dataset[0]["target_sequence"].unsqueeze(0)
    padding_mask = dataset[0]["key_padding_mask"].unsqueeze(0)

    # Forward pass
    out = model(source, padding_mask)
    print("Output shape:", out.shape)
    print("Target shape:", target.shape)
    print("Loss mask shape:", padding_mask.shape)

    # Calculate loss
    loss = cross_entropy(out.transpose(1, 2), target)
    print("Loss:", loss.item())

