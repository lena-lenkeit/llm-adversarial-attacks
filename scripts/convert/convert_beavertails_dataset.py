import argparse

import datasets


def load_and_process_dataset(dataset_path: str):
    # Load the dataset
    dataset = datasets.load_dataset(dataset_path)

    # Concatenate prompt and response
    def concat_fn(prompt: str, response: str):
        return {"text": f"[Prompt]: {prompt} [Response]: {response}"}

    dataset = dataset.map(
        concat_fn,
        input_columns=["prompt", "response"],
        remove_columns=["prompt", "response", "category"],
    )
    dataset = dataset.cast_column("is_safe", datasets.ClassLabel(names=["neg", "pos"]))
    dataset = dataset.rename_column("is_safe", "label")

    dataset = datasets.DatasetDict(
        train=dataset["330k_train"], test=dataset["330k_test"]
    )

    return dataset


def main(args: argparse.Namespace):
    binarized_dataset = load_and_process_dataset(args.src_path)
    binarized_dataset.save_to_disk(args.dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility for processing the Yelp Review dataset."
    )
    parser.add_argument(
        "--src-path",
        type=str,
        default="PKU-Alignment/BeaverTails",
        help="Path to load raw dataset from.",
    )
    parser.add_argument(
        "--dest-path",
        type=str,
        default="datasets/beaver-tails",
        help="Path to write processed dataset to.",
    )

    args = parser.parse_args()
    main(args)
