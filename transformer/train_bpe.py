# %%
from bpe.basic import BasicTokenizer
from datasets import load_dataset

dataset = load_dataset(
    "wmt/wmt14",
    "de-en",
    split='train',
    streaming=True
)


# %%
texts = [ex['translation']['de']
         for ex in dataset] + [ex['translation']['en'] for ex in dataset]

# %%
texts_str = " ".join(texts)


# %%
vocab_size = 37000


# %%
tokenizer = BasicTokenizer()

# %%
tokenizer.train(texts_str, vocab_size, verbose=True)
tokenizer.save('wmt')
