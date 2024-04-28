import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def prepare_data(dataset: datasets.Dataset, layer_id: int):
    features = [row[f"hidden_{layer_id}"] for row in dataset]
    labels = [row[f"label"] for row in dataset]

    features = np.stack(features, axis=0)
    labels = np.stack(labels, axis=0)

    return features, labels


def main():
    # Directories
    activation_path = "activations/pythia-70m-aclimdb"

    # Load dataset
    dataset = datasets.load_from_disk(activation_path)

    # Prepare data
    layer_id = 5

    features_train, labels_train = prepare_data(dataset["train"], layer_id)
    features_test, labels_test = prepare_data(dataset["test"], layer_id)

    # Train probe
    scaler = StandardScaler()
    probe = LogisticRegression()
    pipe = make_pipeline(scaler, probe)

    pipe = pipe.fit(features_train, labels_train)

    # Convert to linear layer
    weight = probe.coef_ / scaler.scale_
    bias = probe.intercept_ - probe.coef_ @ scaler.mean_

    print(weight)
    print(bias)

    # Predict with layer
    pred_labels = features_test @ weight.T + bias

    # Test probe
    plt.figure()
    display = RocCurveDisplay.from_predictions(labels_test, pred_labels, ax=plt.gca())
    plt.show()

    # TODO: Save probe


if __name__ == "__main__":
    main()
