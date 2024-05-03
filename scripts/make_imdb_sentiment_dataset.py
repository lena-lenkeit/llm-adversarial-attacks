import pathlib

import datasets


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
    data_path = "data/aclImdb"
    dataset_path = "datasets/aclImdb"

    dataset = make_imdb_sentiment_dataset(data_path)
    dataset.save_to_disk(dataset_path)


if __name__ == "__main__":
    main()
