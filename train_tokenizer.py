from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

def example_to_text(example):
    # Concatenate the code and docstring with special tokens
    return f"[START] {example['code']} [DOC] {example['docstring']} [END]"

def train_tokenizer(dataset, train_fraction, vocab_size, min_frequency, unk_token, special_tokens, num_workers, tokenizer_file):
    # Load the training data and select a subset to train the tokenizer on
    dataset = load_dataset(dataset)["train"]
    n_subset = int(train_fraction * len(dataset))
    train_data = dataset.select(range(n_subset))
    print(
        f"Loaded dataset of size {len(train_data)} with columns {train_data.column_names}"
    )

    # Combine code and docstrings into single strings
    print("Combining code and docstrings...")
    if num_workers <= 1:
        train_texts = [example_to_text(example) for example in tqdm(train_data)]
    else:
        train_texts = process_map(
            example_to_text,
            train_data,
            max_workers=num_workers,
            chunksize=500,
        )

    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Train the tokenizer
    tokenizer.train_from_iterator(train_texts, trainer=trainer)

    # Save the tokenizer
    tokenizer_path = Path(tokenizer_file)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    print(f"Saved tokenizer to {tokenizer_path}")

if __name__ == "__main__":
    import config
    train_tokenizer(dataset=config.general["dataset"], **config.tokenizer)
