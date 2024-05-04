import os

import datasets
import matplotlib.pyplot as plt
import numpy as np
import safetensors
import safetensors.numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def prepare_data(dataset: datasets.Dataset, layer_id: int):
    dataset = dataset.select_columns([f"hidden_{layer_id}", "label"])
    dataset_dict = dataset.to_dict()

    features = np.stack(dataset_dict[f"hidden_{layer_id}"], axis=0)
    labels = np.stack(dataset_dict["label"], axis=0)

    return features, labels


def main():
    # Directories
    activation_path = "activations/pythia-70m-aclimdb"
    probe_path = "probes/pythia-70m-aclimdb"

    os.makedirs(probe_path, exist_ok=True)

    # Load dataset
    dataset = datasets.load_from_disk(activation_path)

    # Prepare data
    layer_id = 6

    features_train, labels_train = prepare_data(dataset["train"], layer_id)
    features_test, labels_test = prepare_data(dataset["test"], layer_id)

    # Train probe
    scaler = StandardScaler()
    probe = LogisticRegression(max_iter=100000)
    pipe = make_pipeline(scaler, probe)

    pipe = pipe.fit(features_train, labels_train)

    # Convert to linear layer
    # print(probe.coef_.shape)  # (num_classes, num_features)
    # print(probe.intercept_.shape)  # (num_classes,)
    # print(scaler.scale_.shape)  # (num_features,)
    # print(scaler.mean_.shape)  # (num_features,)

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
    RocCurveDisplay.from_predictions(labels_test, scores_linear_test)
    plt.savefig(f"{probe_path}/roc_train.png", bbox_inches="tight")
    plt.savefig(f"{probe_path}/roc_train.svg", bbox_inches="tight")
    plt.close()

    roc_auc_test = roc_auc_score(labels_test, scores_linear_test)
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
    main()
