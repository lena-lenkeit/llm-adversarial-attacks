import argparse

import datasets


def load_and_process_dataset(dataset_path: str, subset: str, train_test_split: float):
    # Load the dataset
    dataset = datasets.load_dataset(dataset_path, subset)
    dataset = dataset["train"]  # Dataset has only one split

    # Filter neutral labels and remap
    dataset = dataset.filter(lambda label: label != 1, input_columns="label")
    dataset = dataset.map(
        lambda label: {"label": {0: 0, 2: 1}[label]}, input_columns="label"
    )
    dataset = dataset.cast_column("label", datasets.ClassLabel(names=["neg", "pos"]))

    # Set correct column name
    dataset = dataset.rename_column("sentence", "text")

    # Add test split
    dataset = dataset.train_test_split(test_size=train_test_split)

    return dataset


def main(args: argparse.Namespace):
    binarized_dataset = load_and_process_dataset(
        args.src_path, args.subset, args.train_test_split
    )
    binarized_dataset.save_to_disk(args.dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility for processing the Financial Phrasebank dataset."
    )
    parser.add_argument(
        "--src-path",
        type=str,
        default="financial_phrasebank",
        help="Path to load raw dataset from.",
    )
    parser.add_argument(
        "--dest-path",
        type=str,
        default="datasets/financial_phrasebank",
        help="Path to write processed dataset to.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="sentences_allagree",
        help="Subset to choose.",
    )
    parser.add_argument(
        "--train-test-split",
        type=float,
        default=0.1,
        help="Size of test set relative to original dataset.",
    )

    args = parser.parse_args()
    main(args)
