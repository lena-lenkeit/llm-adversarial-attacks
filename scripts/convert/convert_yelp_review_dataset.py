import argparse

import datasets


def load_and_process_dataset(dataset_path: str, neg_threshold: int, pos_threshold: int):
    # Load the dataset
    dataset = datasets.load_dataset(dataset_path)

    # Binarize
    def binarize_fn(label: int):
        new_label = -1
        if label >= pos_threshold:
            new_label = 1
        elif label <= neg_threshold:
            new_label = 0

        return {"label": new_label}

    dataset = dataset.map(binarize_fn, input_columns="label")
    dataset = dataset.filter(lambda label: label != -1, input_columns="label")
    dataset = dataset.cast_column("label", datasets.ClassLabel(names=["neg", "pos"]))

    return dataset


def main(args: argparse.Namespace):
    binarized_dataset = load_and_process_dataset(
        args.src_path, args.neg_thresh, args.pos_thresh
    )
    binarized_dataset.save_to_disk(args.dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility for processing the Yelp Review dataset."
    )
    parser.add_argument(
        "--src-path",
        type=str,
        default="yelp_review_full",
        help="Path to load raw dataset from.",
    )
    parser.add_argument(
        "--dest-path",
        type=str,
        default="datasets/yelp_review_full",
        help="Path to write processed dataset to.",
    )
    parser.add_argument(
        "--pos-thresh",
        type=int,
        default=4,
        help="Ratings at or above this threshold will be classified as positive.",
    )
    parser.add_argument(
        "--neg-thresh",
        type=int,
        default=2,
        help="Ratings at or below this threshold will be classified as negative.",
    )

    args = parser.parse_args()
    main(args)
