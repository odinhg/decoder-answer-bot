import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class QADataset(Dataset):
    def __init__(
        self, config, tokenizer, 
    ):
        self.dataset = load_dataset(dataset_name)[split]
        n_subset = int(fraction * len(self.hf_dataset))
        self.hf_dataset = self.hf_dataset.select(range(n_subset))
        print(
            f"Loaded dataset of size {len(self.hf_dataset)} with columns {self.hf_dataset.column_names}"
        )

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_code = ignore_code

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        code, docstring = (
            self.hf_dataset[idx]["code"],
            self.hf_dataset[idx]["docstring"],
        )

        max_length_each = (self.max_length - 3) // 2
        pad_id = self.tokenizer.token_to_id("[PAD]")
        doc_id = self.tokenizer.token_to_id("[DOC]")
        start_id = self.tokenizer.token_to_id("[START]")
        end_id = self.tokenizer.token_to_id("[END]")

        # Tokenize the code and docstring
        code_ids = self.tokenizer.encode(code).ids
        docstring_ids = self.tokenizer.encode(docstring).ids

        # Truncate if the combined length is greater than max_length (equally truncate from both code and docstring)
        code_ids = code_ids[:max_length_each]
        docstring_ids = docstring_ids[:max_length_each]

        # Combine the tokens with special tokens
        tokenized_sequence = [start_id] + code_ids + [doc_id] + docstring_ids + [end_id]

        # Pad to max_length (if needed)
        pad_length = self.max_length - len(tokenized_sequence) + 1
        tokenized_sequence += [pad_id] * pad_length
        tokenized_sequence = torch.tensor(tokenized_sequence)
        
        # Create source and target sequences (shifted by one)
        source_sequence = tokenized_sequence[:-1].clone()
        target_sequence = tokenized_sequence[1:].clone()

        # Masking padding tokens (True = padding)
        key_padding_mask = torch.zeros_like(source_sequence)
        key_padding_mask[-pad_length:] = 1
        key_padding_mask = key_padding_mask.bool()

        # Loss mask
        loss_mask = torch.zeros_like(target_sequence)
        loss_mask[-pad_length:] = 1
        if self.ignore_code:
            loss_mask[0:len(code_ids) + 1] = 1
        loss_mask = loss_mask.bool()
        target_sequence[loss_mask] = -100

        return {
            "source_sequence": source_sequence,
            "target_sequence": target_sequence,
            "key_padding_mask": key_padding_mask,
        }


if __name__ == "__main__":
    import config
    from tokenizers import Tokenizer
    from datasets import load_dataset

    tokenizer = Tokenizer.from_file(config.tokenizer["tokenizer_file"])
    idx = 1
    max_length = 32
    code_doc_dataset = CodeDocstringDataset(
        config.general["dataset"], tokenizer, max_length
    )

    source_sequence, target_sequence, key_padding_mask = code_doc_dataset[idx].values()

    print("Source sequence shape:", source_sequence.shape)
    print("Target sequence shape:", target_sequence.shape)
    print("Key padding mask shape:", key_padding_mask.shape)

    print("Source sequence:", source_sequence)
    print("Target sequence:", target_sequence)
    print("Key padding mask:", key_padding_mask)

