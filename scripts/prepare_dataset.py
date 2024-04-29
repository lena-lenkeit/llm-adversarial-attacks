import pathlib

import datasets
import numpy as np
import torch
from tqdm.auto import tqdm as tq
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast


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


@torch.no_grad()
def main():
    # Parameters
    max_elements = 256
    max_length = 512
    device = "cuda"

    # Directories
    model_path = "EleutherAI/pythia-70m"
    dataset_path = "data/aclImdb"
    activation_cache_path = "activations/pythia-70m-aclimdb"

    # Load model, tokenizer and dataset
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device, torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = make_imdb_sentiment_dataset(dataset_path)

    model.to_bettertransformer()
    model.eval()

    # Limit dataset length
    for split in dataset:
        dataset[split] = dataset[split].shuffle(1234, keep_in_memory=True)
        dataset[split] = dataset[split].flatten_indices(keep_in_memory=True)
        dataset[split] = dataset[split].select(range(max_elements), keep_in_memory=True)

    # Print info on model, tokenizer and dataset
    print("---MODEL---")
    print(model)
    print("---TOKENIZER---")
    print(tokenizer)
    print("---DATASET---")
    print(dataset)

    # Run entire dataset through model, caching all hidden layer activations at the last
    # token
    for split in dataset:
        activation_cache = {}
        for row in tq(dataset[split]):
            text: str = row["text"]

            tokens = tokenizer(
                text, max_length=max_length, truncation=True, return_tensors="pt"
            )
            tokens = tokens.to(device)

            with torch.no_grad():
                outputs: CausalLMOutputWithPast = model(
                    **tokens, output_hidden_states=True
                )

            for cache_id, hidden_state in enumerate(outputs.hidden_states):
                cache = activation_cache.get(cache_id, [])
                cache.append(hidden_state[0, -1].cpu().numpy())
                activation_cache[cache_id] = cache

        for cache_id in activation_cache:
            dataset[split] = dataset[split].add_column(
                f"hidden_{cache_id}", activation_cache[cache_id]
            )

    # Save dataset, with activations included
    dataset.save_to_disk(activation_cache_path)


if __name__ == "__main__":
    main()
