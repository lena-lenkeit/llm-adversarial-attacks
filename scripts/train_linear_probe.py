import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import safetensors
import safetensors.numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import datasets


def prepare_data(dataset: datasets.Dataset, layer_id: int):
    dataset = dataset.select_columns([f"hidden_{layer_id}", "label"])
    dataset_dict = dataset.to_dict()

    features = np.stack(dataset_dict[f"hidden_{layer_id}"], axis=0)
    labels = np.stack(dataset_dict["label"], axis=0)

    return features, labels


def main(args: argparse.Namespace):
    # Directories
    activation_path = args.activation_path
    probe_path = args.probe_path

    os.makedirs(probe_path, exist_ok=True)

    # Load dataset
    dataset = datasets.load_from_disk(activation_path)

    # Prepare data
    layer_id = args.layer_id

    features_train, labels_train = prepare_data(dataset["train"], layer_id)
    features_test, labels_test = prepare_data(dataset["test"], layer_id)

    # Train probe
    scaler = StandardScaler()
    probe = LogisticRegression(max_iter=args.max_iter)
    pipe = make_pipeline(scaler, probe)

    pipe = pipe.fit(features_train, labels_train)

    # Convert to linear layer
    weight = probe.coef_ / scaler.scale_[None]  # (num_classes, num_features)
    bias = probe.intercept_ - (scaler.mean_[None] @ weight.T)[0]  # (num_classes,)

    # Predict with layer
    scores_pipe_train = pipe.decision_function(features_train)
    scores_linear_train = features_train @ weight.T + bias

    scores_pipe_test = pipe.decision_function(features_test)
    scores_linear_test = features_test @ weight.T + bias

    assert np.isclose(scores_pipe_train[:, None], scores_linear_train).all()
    assert np.isclose(scores_pipe_test[:, None], scores_linear_test).all()

    # Test probe
    roc_auc_train = roc_auc_score(labels_train, scores_linear_train)
    roc_auc_test = roc_auc_score(labels_test, scores_linear_test)

    if args.plot:
        RocCurveDisplay.from_predictions(labels_test, scores_linear_test)
        plt.savefig(f"{probe_path}/roc_train.png", bbox_inches="tight")
        plt.savefig(f"{probe_path}/roc_train.svg", bbox_inches="tight")
        plt.close()

        RocCurveDisplay.from_predictions(labels_test, scores_linear_test)
        plt.savefig(f"{probe_path}/roc_test.png", bbox_inches="tight")
        plt.savefig(f"{probe_path}/roc_test.svg", bbox_inches="tight")
        plt.close()

    # Save probe
    safetensors.numpy.save_file(
        {"weight": weight, "bias": bias},
        f"{probe_path}/probe.safetensors",
        metadata={
            "activation_path": activation_path,
            "layer_id": str(layer_id),
            "roc_auc_train": f"{roc_auc_train:.4f}",
            "roc_auc_test": f"{roc_auc_test:.4f}",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a probe.")
    parser.add_argument(
        "--activation_path",
        type=str,
        default="activations/pythia-160m-coherence",
        help="Path to the activation dataset.",
    )
    parser.add_argument(
        "--probe_path",
        type=str,
        default="probes/pythia-160m-coherence",
        help="Path to save the probe to.",
    )
    parser.add_argument(
        "--layer_id",
        type=int,
        default=None,
        help="Hidden layer ID to extract features from.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000000,
        help="Maximum number of iterations for the logistic regression to converge.",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Set to plot ROC curves.",
    )

    args = parser.parse_args()
    main(args)
