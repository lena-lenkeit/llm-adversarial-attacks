import argparse

import datasets


def load_and_process_dataset(dataset_path: str):
    # Load the dataset
    dataset = datasets.load_dataset(dataset_path)

    # Merge validation and test splits
    dataset = datasets.DatasetDict(
        train=dataset["train"],
        test=datasets.concatenate_datasets([dataset["validation"], dataset["test"]]),
    )

    return dataset


def main(args: argparse.Namespace):
    binarized_dataset = load_and_process_dataset(args.src_path)
    binarized_dataset.save_to_disk(args.dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility for processing the Rotten Tomatoes dataset."
    )
    parser.add_argument(
        "--src-path",
        type=str,
        default="rotten_tomatoes",
        help="Path to load raw dataset from.",
    )
    parser.add_argument(
        "--dest-path",
        type=str,
        default="datasets/rotten_tomatoes",
        help="Path to write processed dataset to.",
    )

    args = parser.parse_args()
    main(args)
