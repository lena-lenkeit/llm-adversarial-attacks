import argparse

import datasets


def load_and_process_dataset(dataset_path: str):
    # Load the dataset
    dataset = datasets.load_dataset(dataset_path)

    # Select correct splits
    dataset = datasets.DatasetDict(
        train=dataset["train"],
        test=dataset["test"],
    )

    return dataset


def main(args: argparse.Namespace):
    binarized_dataset = load_and_process_dataset(args.src_path)
    binarized_dataset.save_to_disk(args.dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility for processing the StanfordNLP IMDB dataset."
    )
    parser.add_argument(
        "--src-path",
        type=str,
        default="stanfordnlp/imdb",
        help="Path to load raw dataset from.",
    )
    parser.add_argument(
        "--dest-path",
        type=str,
        default="datasets/imdb",
        help="Path to write processed dataset to.",
    )

    args = parser.parse_args()
    main(args)
