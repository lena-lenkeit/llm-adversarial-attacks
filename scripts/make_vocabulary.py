import itertools
import json
import os
from collections import Counter

from transformers import AutoTokenizer

import datasets


def main():
    # Directories
    tokenizer_path = "EleutherAI/pythia-1b"
    dataset_path = "datasets/aclImdb"
    dictionary_path = "dictionaries/aclimdb"

    # Parameters
    max_length = 512
    truncation = False

    os.makedirs(dictionary_path, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load dataset
    dataset = datasets.load_from_disk(dataset_path)
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
