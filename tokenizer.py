from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

def example_to_text(example):
    return f"[QST]{example['question']}[ANS]{example['answer']}[END]"

def train_tokenizer(config):
    # Load the training data and select a subset to train the tokenizer on
    dataset = load_dataset(config.general.dataset)[config.general.split]
    n_subset = int(config.tokenizer.train_fraction * len(dataset))
    train_data = dataset.select(range(n_subset))
    print(
        f"Loaded dataset of size {len(train_data)} with columns {train_data.column_names}"
    )

    # Combine questions and answers into single strings 
    print("Combining strings...")
    if config.tokenizer.num_workers <= 1:
        train_texts = [example_to_text(example) for example in tqdm(train_data)]
    else:
        train_texts = process_map(
            example_to_text,
            train_data,
            max_workers=config.tokenizer.num_workers,
            chunksize=500,
        )

    tokenizer = Tokenizer(models.BPE(unk_token=config.tokenizer.unk_token))
    tokenizer.normalizer = normalizers.BertNormalizer(clean_text=True, strip_accents=True)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=config.tokenizer.vocab_size,
        min_frequency=config.tokenizer.min_frequency,
        special_tokens=config.tokenizer.special_tokens,
        show_progress=True,
    )

    # Train the tokenizer
    tokenizer.train_from_iterator(train_texts, trainer=trainer)

    # Save the tokenizer
    tokenizer_path = Path(config.tokenizer.tokenizer_filename)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    print(f"Saved tokenizer to {tokenizer_path}")

    return tokenizer

if __name__ == "__main__":
    from config import config

    if not Path(config.tokenizer.tokenizer_filename).exists():
        tokenizer = train_tokenizer(config)
    else:
        print(f"Using existing tokenizer at {config.tokenizer.tokenizer_filename}")
        tokenizer = Tokenizer.from_file(config.tokenizer.tokenizer_filename)

    # Simple test
    input_question = "Who is the president of the United States?"
    input_answer = "That would be Donald Musk, I believe."
    example = {"question": input_question, "answer": input_answer}
    text = example_to_text(example)
    print(f"Tokenizer Input: {text}")
    encoded = tokenizer.encode(text)
    print(f"Encoded: {encoded.ids}")
    decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
    print(f"Decoded: {decoded}")


