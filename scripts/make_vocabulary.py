import itertools
import json
import os
import pathlib
from collections import Counter

import datasets
from transformers import AutoTokenizer


def make_imdb_sentiment_dataset(dataset_root_dir: str):
    def load_split(split: str):
        ids = []
        scores = []
        labels = []
        texts = []

        filepaths = pathlib.Path(dataset_root_dir).glob(f"{split}/*/*.txt")
        for filepath in filepaths:
            label = filepath.parent.name
            if label == "unsup":
                continue

            id, score = [int(s) for s in filepath.name.split(".")[0].split("_")]

            with open(filepath, mode="r") as f:
                text = f.read()

            ids.append(id)
            scores.append(score)
            labels.append(label)
            texts.append(text)

        return {"id": ids, "score": scores, "label": labels, "text": texts}

    train_dataset = datasets.Dataset.from_dict(load_split("train"))
    test_dataset = datasets.Dataset.from_dict(load_split("test"))

    train_dataset = train_dataset.class_encode_column("label")
    test_dataset = test_dataset.class_encode_column("label")

    label_mapping = {"pos": 1, "neg": 0}
    train_dataset = train_dataset.align_labels_with_mapping(label_mapping, "label")
    test_dataset = test_dataset.align_labels_with_mapping(label_mapping, "label")

    dataset = datasets.DatasetDict()
    dataset["train"] = train_dataset
    dataset["test"] = test_dataset

    return dataset


def main():
    # Directories
    tokenizer_path = "EleutherAI/pythia-1b"
    dataset_path = "data/aclImdb"
    dictionary_path = "dictionaries/aclimdb"

    # Parameters
    max_length = 512
    truncation = False

    os.makedirs(dictionary_path, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load dataset
    dataset = make_imdb_sentiment_dataset(dataset_path)
    dataset = dataset["train"]

    # Tokenize
    dataset = dataset.select_columns("text")
    dataset = dataset.map(
        lambda text: {
            "token_ids": tokenizer(
                text, max_length=max_length, truncation=truncation
            ).input_ids
        },
        batched=True,
        input_columns="text",
        remove_columns="text",
    )

    # Count
    token_ids = dataset.to_dict()["token_ids"]
    counter = Counter(itertools.chain(*token_ids))
    sorted_counts = {
        k: v for k, v in sorted(counter.items(), key=lambda x: x[1], reverse=True)
    }

    # Save
    with open(f"{dictionary_path}/counts_ids.json", mode="w") as f:
        json.dump(sorted_counts, f, indent=4)

    with open(f"{dictionary_path}/counts_tokens.json", mode="w") as f:
        json.dump(
            {tokenizer.convert_ids_to_tokens(k): v for k, v in sorted_counts.items()},
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()
