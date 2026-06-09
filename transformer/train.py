from datasets import load_dataset

dataset = load_dataset(
    "wmt/wmt14",
    "de-en"
)

print(dataset['train'][0])