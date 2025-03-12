# TODO

- [ ] Fix optimizer state dict device when loading existing
- [ ] Make config cleaner (maybe use simple namespace)
- [ ] Save model and optimizer state dict multiple times per epoch in case Google Colab times out. This way we can resume training from the last checkpoint when we have available resources again.
- [x] Config for one small and one larger model
- [x] Create source and target sequences in dataset class (shifted)
- [x] Move to PyTorch masked attention layer instead of custom (for performance)
- [x] Change attention mask generation so it is comatible with PyTorch's attention layer
- [x] Instead of returning loss mask, just return target sequence with padding tokens set to -100 (default `ignore_index` in CrossEntropyLoss)
- [ ] Add special tokens for indentation (4 spaces) and newlines in code.
- [x] Include code tokens in loss and see if it improves performance
- [ ] Implement beam search or other non-trivial decoding strategy

# Py2Doc – Generating Documentation Strings from Python Code

This is a basic implementation of an auto-regressive decoder-only transformer model for generating documentation strings (docstrings) from Python code. It uses a masked self-attention layer (causal attention), implemented from scratch in PyTorch.

The input format to the model is on the form `[START] c1 c2 ... cN [DOC] d1 d2 ... dM [END] [PAD] ... [PAD]` where `c1 c2 ... cN` is the tokenized code and `d1 d2 ... dM` is the tokenized docstring (truncated and/or padded to a fixed maximum length if necessary). 

## Quick Start

0. Set parameters to use in `config.py`.
1. Run `train_tokenizer.py` to train the tokenizer.
2. Run `train_transformer.py` to train the transformer model.
3. Run `generate_docstrings.py` to generate docstrings for some example code snippets.
4. Run `webapp.py` to start a simple web interface for generating docstrings.

### Tokenizer

The tokenizer is trained using Byte Pair Encoding (BPE). The training script is in `train_tokenizer.py`. The tokenizer is saved as a JSON file.

**Special tokens:**

- `[START]` is used as the start token.
- `[DOC]` is used between the code and the docstring.
- `[END]` is used as the end token.
- `[PAD]` is used as padding.

### Dataset

The dataset class `CodeDocstringDataset` is responsible for providing training examples to the model. It loads a given dataset from Hugging Face's datasets library and tokenizes the code and docstrings using the trained tokenizer. It also takes care of the truncation and padding of sequences, as well as the creation of the attention and loss mask tensors (probably not optimal memory-wise but makes the data loading self-contained).

### Masked Self-Attention (Causal Attention)

The masked self-attention layer is implemented in `masked_self_attention.py`. It is used in the transformer model and requires a causal mask to prevent the model from looking ahead in the sequence. Padding tokens are also masked out to prevent the model from attending to them.

### Transformer Model

The transformer model is implemented in `transformer.py`. It consists of a stack of decoder layers, each with a masked self-attention layer and a feed-forward neural network.

To train the model using the training script `train_transformer.py`. The model is trained using the Adam optimizer and the cross-entropy loss function. All code and padding tokens are ignored when calculating the loss.

### Inference

Todo

### Questions

**1. What are the practical implications of increasing the maximum sequence length?**

**2. How does the vocabulary size affect the model size and training time?**

**3. What are the benefits and drawbacks of using beam search over greedy decoding for inference?**

**4. What is meant by "auto-regressive" in the context of transformer models?**

**5. How does the transformer model handle variable-length sequences?**

**6. What is the role of masking in the self-attention layers?**

**7. What is the role of masking the padding tokens in the attention layers?**

**8. What is the role of applying masking to the cross-entropy loss function?**

**9. How many parameters does your model have? Compare it to the GPT-1, GPT-2, and GPT-3 models.**

## Learning Resources


- [Decoder-Only Transformers: The Workhorse of Generative LLMs (Blog post)](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse)
- [How does the (decoder-only) transformer architecture work? (AI StackExchange)](https://ai.stackexchange.com/a/40180)
- [Attention in transformers, step-by-step (3Blue1Brown, YouTube)](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- [Attention is all you need (Original transformers paper)](https://arxiv.org/pdf/1706.03762)
- [Stack Overflow answer explaining the role of masking in attention layers](https://stackoverflow.com/a/59713254)

---

I made this project for the course "Deep Learning (INF265)" at the University of Bergen (UiB) in the spring of 2025.

