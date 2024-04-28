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

    dataset = datasets.DatasetDict()
    dataset["train"] = train_dataset
    dataset["test"] = test_dataset

    return dataset


def main():
    dataset = make_imdb_sentiment_dataset("data/aclImdb")
    print(dataset)


if __name__ == "__main__":
    main()
