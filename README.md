# My Little Language Model (MLLM)

This is a from-scratch implementation of a decoder-only transformer model for generating answers to short questions. It uses masked (causal) self-attention layers to prevent the model from looking ahead in the sequence. It supports inference using either greedy decoding or top-p (nucleus) sampling.

The sequence format is as follows:

```
[QUESTION] q1 q2 ... qN [ANSWER] a1 a2 ... aM [END]
```

where `[QUESTION]` and `[ANSWER]` are special tokens indicating the start of the question and answer sequences, respectively, and `[END]` is a special token indicating the end of the sequence.

The dataset is a subset of the [GooAQ dataset](https://github.com/allenai/gooaq). Here are a few examples:

```
Q: is it possible to get a false negative flu test?
A: This variation in ability to detect viruses can result in some people who are infected with the flu having a negative rapid test result. (This situation is called a false negative test result.)
```

```
Q: are you not supposed to rinse after brushing teeth?
A: Don't rinse with water straight after toothbrushing Don't rinse your mouth immediately after brushing, as it'll wash away the concentrated fluoride in the remaining toothpaste. This dilutes it and reduces its preventative effects.
```

```
Q: what is the difference between a bald eagle and a hawk?
A: Hawks have curved beak and very sharp talons. Legs of both eagles and hawks are at least partially covered with feathers. Eagles have a wingspan of 8 feet, while most hawks have a wingspan of less than 5 feet. Hawks can soar for long period of time thanks to their long, broad wings and wide tail.
```

We always keep the full question but allow truncation of the answer to keep the sequence length manageable for training on low-cost hardware.

### Masked Self-Attention (Causal Attention)

The masked self-attention layer is implemented in `masked_self_attention.py`. It is used in the transformer model and requires a causal mask to prevent the model from looking ahead in the sequence. Padding tokens are also masked out to prevent the model from attending to them.

### Transformer Model

The transformer model is implemented in `transformer.py`. It consists of a stack of decoder layers, each with a masked self-attention layer and a feed-forward neural network.

To train the model using the training script `train_transformer.py`. The model is trained using the Adam optimizer and the cross-entropy loss function. All code and padding tokens are ignored when calculating the loss.

### Inference

Currently, two sampling strategies are implemented: Greedy and Top-p (nucleus) sampling. The inference script `generate_answers.py -q <question>` generates an answer to the given question using the trained model. Furthermore, running `web_app.py` will start a web application where you can input questions and get answers interactively.

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

**10. Would an encoder-decoder transformer model be more suitable for this task? What are the benefits and drawbacks?**

## Learning Resources

Some websites and videos that are helpful for understanding transformers and self-attention:

- [Decoder-Only Transformers: The Workhorse of Generative LLMs (Blog post)](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse)
- [How does the (decoder-only) transformer architecture work? (AI StackExchange)](https://ai.stackexchange.com/a/40180)
- [Attention in transformers, step-by-step (3Blue1Brown, YouTube)](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- [Attention is all you need (Original transformers paper)](https://arxiv.org/pdf/1706.03762)
- [Stack Overflow answer explaining the role of masking in attention layers](https://stackoverflow.com/a/59713254)

---

I made this project for the course "Deep Learning (INF265)" at the University of Bergen (UiB) in the spring of 2025.

