# My Little Language Model (aka the cursed chatbot)

![screenshot of chatbot interface](figs/chatbot_screenshot.png)

This is a from-scratch implementation of a decoder-only transformer model for generating answers to short questions. 

- **Decoder-only** transformer model for **causal language modeling** 
- **Masked self-attention** layer for causal attention
- Only **35M parameters**
- BPE Tokenizer trained from scratch with a **vocabulary size of 20k**
- Trained on a subset of the GooAQ dataset with ~**800k question-answer pairs** (no pre-training)
- Supports **greedy** and **top-p** (nucleus) sampling at inference time
- Super basic **chatbot interface** for interacting with the model based on streamlit

## Quickstart

1. Use the notebook `gpu_training_colab_notebook.ipynb` that can be used to train the model on Google Colab. 
2. Download the model checkpoint and tokenizer JSON file and put them in the `temp` directory.
3. Run `streamlit run chatbot.py` to start the chatbot interface.
4. Get your questions answered by the wackiest chatbot you've ever seen!

## Data Format

The sequence format is as follows:

```
[QUESTION] q1 q2 ... qN [ANSWER] a1 a2 ... aM [END]
```

where `[QUESTION]` and `[ANSWER]` are special tokens indicating the start of the question and answer sequences, respectively, and `[END]` is a special token indicating the end of the sequence.

The dataset is a subset of the [GooAQ dataset](https://github.com/allenai/gooaq). 

**Example questions and answers:**
```
Q: is it possible to get a false negative flu test?
A: This variation in ability to detect viruses can result in some people who are infected with the flu having a negative rapid test result. (This situation is called a false negative test result.)
```

```
Q: are you not supposed to rinse after brushing teeth?
A: Don't rinse with water straight after toothbrushing Don't rinse your mouth immediately after brushing, as it'll wash away the concentrated fluoride in the remaining toothpaste. This dilutes it and reduces its preventative effects.
```

## Learn More

### Going Further

Here are some ideas for extending the project:

- **Pre-training**: Pre-train the model on a large corpus of text data to improve performance.
- **Fine-tuning**: Fine-tune the model on the question-answering task prioritizing answer-generation.
- **Hyperparameter tuning**: Experiment with different hyperparameters to improve performance.
- **Scaling up**: Train a larger model with more parameters and a larger dataset.

### Learning Resources

Some websites and videos that are helpful for understanding transformers and self-attention:

- [Decoder-Only Transformers: The Workhorse of Generative LLMs (Blog post)](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse)
- [How does the (decoder-only) transformer architecture work? (AI StackExchange)](https://ai.stackexchange.com/a/40180)
- [Attention in transformers, step-by-step (3Blue1Brown, YouTube)](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- [Attention is all you need (Original transformers paper)](https://arxiv.org/pdf/1706.03762)
- [Stack Overflow answer explaining the role of masking in attention layers](https://stackoverflow.com/a/59713254)

### Questions

Here are some natural questions that arise from this project that you might want to think about:

**1. What are the practical implications of increasing the maximum sequence length?**

**2. How does the vocabulary size affect the model size and training time?**

**3. What are the advantages and drawbacks of different sampling strategies (beam search, top-p and greedy)?** 

**4. What is meant by "auto-regressive" in the context of transformer models?**

**5. How does the transformer model handle variable-length sequences?**

**6. What is the role of masking in the self-attention layers?**

**7. What is the role of masking the padding tokens in the attention layers?**

**8. What is the role of applying masking to the cross-entropy loss function (i.e., setting some labels to `-100`)?**

**9. How many parameters does your model have? Compare it to the GPT-1, GPT-2, and GPT-3 models.**

**10. Would an encoder-decoder transformer model be more suitable for this task? What are the benefits and drawbacks?**

---

I made this project for the course "Deep Learning (INF265)" at the University of Bergen (UiB) in the spring of 2025.

