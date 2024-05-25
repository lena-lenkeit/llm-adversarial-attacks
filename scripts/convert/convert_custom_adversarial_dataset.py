import argparse
import json

import datasets


def load_token_strings(path: str):
    with open(path, mode="r") as f:
        data = json.load(f)

    tokens = data["adv_tokens"]
    valid = data["roundtrip_full_check"]

    tokens = [t for t, v in zip(tokens, valid) if v]
    return tokens


def load_and_process_dataset(pos_path: str, neg_path: str, train_test_split: float):
    # Load tokens
    pos_tokens = load_token_strings(pos_path)
    neg_tokens = load_token_strings(neg_path)

    all_tokens = []
    all_tokens.extend(pos_tokens)
    all_tokens.extend(neg_tokens)

    # Prepare labels
    labels = [1] * len(pos_tokens) + [0] * len(neg_tokens)

    # To dataset
    dataset = datasets.Dataset.from_dict({"text": all_tokens, "label": labels})
    dataset = dataset.cast_column("label", datasets.ClassLabel(names=["neg", "pos"]))

    # Add test split
    dataset = dataset.train_test_split(test_size=train_test_split)

    return dataset


def main(args: argparse.Namespace):
    dataset = load_and_process_dataset(
        args.src_pos_path, args.src_neg_path, args.train_test_split
    )
    dataset.save_to_disk(args.dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility for combining two adversarial token files into a dataset."
    )
    parser.add_argument(
        "--src-pos-path",
        type=str,
        required=True,
        help="Path to .json to load positive strings from.",
    )
    parser.add_argument(
        "--src-neg-path",
        type=str,
        required=True,
        help="Path to .json to load negative strings from.",
    )
    parser.add_argument(
        "--dest-path",
        type=str,
        required=True,
        help="Path to write processed dataset to.",
    )
    parser.add_argument(
        "--train-test-split",
        type=float,
        default=0.1,
        help="Size of test set relative to original dataset.",
    )

    args = parser.parse_args()
    main(args)
