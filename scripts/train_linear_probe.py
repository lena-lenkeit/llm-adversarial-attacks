import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Literal

import matplotlib.pyplot as plt
import numpy as np
import safetensors
import safetensors.numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm as tq

import datasets


def prepare_data(dataset: datasets.Dataset, layer_id: int):
    dataset = dataset.select_columns([f"hidden_{layer_id}", "label"])
    dataset_dict = dataset.to_dict()

    features = np.stack(dataset_dict[f"hidden_{layer_id}"], axis=0)
    labels = np.stack(dataset_dict["label"], axis=0)

    return features, labels


def get_num_layers(dataset: datasets.Dataset):
    return len([name for name in dataset.column_names if name.startswith("hidden_")])


@dataclass
class ProbeMetrics:
    logits: np.ndarray
    targets: np.ndarray
    roc_auc: float


@dataclass
class ProbeEvals:
    eval_layer_id: int
    train_metrics: ProbeMetrics
    test_metrics: ProbeMetrics


@dataclass
class Probe:
    weight: np.ndarray
    bias: np.ndarray
    train_layer_id: int


def train_probe(dataset: datasets.DatasetDict, layer_id: int, max_iter: int):
    features_train, labels_train = prepare_data(dataset["train"], layer_id)
    features_test, labels_test = prepare_data(dataset["test"], layer_id)

    # Train probe
    scaler = StandardScaler()
    probe = LogisticRegression(max_iter=max_iter)
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

    # Validate conversion
    assert np.isclose(scores_pipe_train[:, None], scores_linear_train).all()
    assert np.isclose(scores_pipe_test[:, None], scores_linear_test).all()

    # Test probe
    roc_auc_train = roc_auc_score(labels_train, scores_linear_train)
    roc_auc_test = roc_auc_score(labels_test, scores_linear_test)

    return Probe(
        weight,
        bias,
        layer_id,
    ), ProbeEvals(
        layer_id,
        train_metrics=ProbeMetrics(scores_linear_train, labels_train, roc_auc_train),
        test_metrics=ProbeMetrics(scores_linear_test, labels_test, roc_auc_test),
    )


def save_probe(probe: Probe, probe_path: str, metadata: Dict[str, str] = {}):
    data = {
        "weight": probe.weight,
        "bias": probe.bias,
        "train_layer_id": np.int64(probe.train_layer_id),
    }

    os.makedirs(probe_path, exist_ok=True)
    safetensors.numpy.save_file(
        data, f"{probe_path}/probe.safetensors", metadata=metadata
    )


def save_eval(probe_eval: ProbeEvals, probe_path: str, metadata: Dict[str, str] = {}):
    data = {
        "layer_id": np.int64(probe_eval.eval_layer_id),
        "train_logits": probe_eval.train_metrics.logits,
        "train_targets": probe_eval.train_metrics.targets,
        "train_roc_auc": np.float64(probe_eval.train_metrics.roc_auc),
        "test_logits": probe_eval.test_metrics.logits,
        "test_targets": probe_eval.test_metrics.targets,
        "test_roc_auc": np.float64(probe_eval.test_metrics.roc_auc),
    }

    os.makedirs(probe_path, exist_ok=True)
    safetensors.numpy.save_file(
        data, f"{probe_path}/eval.safetensors", metadata=metadata
    )


def get_metric(
    evals: ProbeEvals, metric: Literal["train_roc_auc", "test_roc_auc"]
) -> float:
    if metric == "train_roc_auc":
        return evals.train_metrics.roc_auc
    elif metric == "test_roc_auc":
        return evals.test_metrics.roc_auc


def main(args: argparse.Namespace):
    # Load dataset
    dataset = datasets.load_from_disk(args.activation_path)
    num_layers = get_num_layers(dataset["train"])

    # Train probes
    if args.layer_id is None:
        layer_ids = list(range(num_layers))
    else:
        layer_ids = args.layer_id

    probes: List[Probe] = []
    evals: List[ProbeEvals] = []
    for layer_id in tq(layer_ids):
        probe, probe_eval = train_probe(dataset, layer_id, args.max_iter)
        probes.append(probe)
        evals.append(probe_eval)

    # Find best probe
    best_id = 0
    best_roc_auc = get_metric(evals[0], args.best_metric)

    for i, probe_evals in enumerate(evals[1:], start=1):
        probe_roc_auc = get_metric(probe_evals, args.best_metric)
        if probe_roc_auc > best_roc_auc:
            best_id = i
            best_roc_auc = probe_roc_auc

    # Save run info
    os.makedirs(args.probe_path, exist_ok=True)
    with open(f"{args.probe_path}/info.json", mode="w") as f:
        info = {
            "activation_path": args.activation_path,
            "best_layer_id": probes[best_id].train_layer_id,
            "best_test_roc_auc": best_roc_auc,
        }

        json.dump(info, f, indent=4)

    # Save probes
    save_probe(probes[best_id], f"{args.probe_path}/best")
    save_eval(evals[best_id], f"{args.probe_path}/best")

    for probe, probe_eval in zip(probes, evals):
        save_probe(probe, f"{args.probe_path}/layer_{probe.train_layer_id}")
        save_eval(probe_eval, f"{args.probe_path}/layer_{probe.train_layer_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a probe.")
    parser.add_argument(
        "--activation_path",
        type=str,
        default="activations/pythia-70m-aclimdb",
        help="Path to the activation dataset.",
    )
    parser.add_argument(
        "--probe_path",
        type=str,
        default="probes/pythia-70m-aclimdb-v2",
        help="Path to save the probe to.",
    )
    parser.add_argument(
        "--layer_id",
        type=int,
        nargs="*",
        default=None,
        help="Hidden layer ID(s) to extract features from.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000000,
        help="Maximum number of iterations for the logistic regression to converge.",
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        choices=["train_roc_auc", "test_roc_auc"],
        default="train_roc_auc",
        help="Metric to use to select the best probe among all probes.",
    )

    args = parser.parse_args()
    main(args)
