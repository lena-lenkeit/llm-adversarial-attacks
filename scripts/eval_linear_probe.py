import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict

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
    features = np.stack(dataset[f"hidden_{layer_id}"], axis=0)
    labels = np.stack(dataset["label"], axis=0)

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


def eval_probe(
    dataset: datasets.DatasetDict, probe: Probe, layer_id: int | None = None
):
    # Prepare data
    layer_id = probe.train_layer_id if layer_id is None else layer_id
    features_train, labels_train = prepare_data(dataset["train"], layer_id)
    features_test, labels_test = prepare_data(dataset["test"], layer_id)

    # Predict with probe
    probe_logits_train = features_train @ probe.weight.T + probe.bias
    probe_logits_test = features_test @ probe.weight.T + probe.bias

    # Get ROC AUC scores
    roc_auc_train = roc_auc_score(labels_train, probe_logits_train)
    roc_auc_test = roc_auc_score(labels_test, probe_logits_test)

    return ProbeEvals(
        eval_layer_id=layer_id,
        train_metrics=ProbeMetrics(
            logits=probe_logits_train,
            targets=labels_train,
            roc_auc=roc_auc_train,
        ),
        test_metrics=ProbeMetrics(
            logits=probe_logits_test,
            targets=labels_test,
            roc_auc=roc_auc_test,
        ),
    )


def load_probe(probe_path: str):
    probe_data = safetensors.numpy.load_file(f"{probe_path}/probe.safetensors")
    return Probe(
        weight=probe_data["weight"],
        bias=probe_data["bias"],
        train_layer_id=probe_data["train_layer_id"],
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


def main(args: argparse.Namespace):
    # Load dataset
    dataset = datasets.load_from_disk(args.activation_path)

    # Expand arguments
    num_probes = len(args.probe_path)

    if args.layer_id is None:
        layer_ids = [None] * num_probes
    elif len(args.layer_id) == 1:
        layer_ids = args.layer_id * num_probes
    else:
        layer_ids = args.layer_id

    if len(args.eval_path) == 1:
        eval_paths = args.eval_path * num_probes
    else:
        eval_paths = args.eval_path

    # Evaluate probes
    for probe_path, eval_path, layer_id in zip(args.probe_path, eval_paths, layer_ids):
        probe = load_probe(probe_path)
        probe_eval = eval_probe(dataset, probe, layer_id)
        save_eval(probe_eval, eval_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained probe or set of probes on a different dataset."
    )
    parser.add_argument(
        "--activation_path",
        type=str,
        default="activations/pythia-70m-aclimdb",
        help="Path to the activation dataset.",
    )
    parser.add_argument(
        "--probe_path",
        type=str,
        nargs="+",
        default=[
            "probes/pythia-70m-aclimdb-v2/best",
            "probes/pythia-70m-aclimdb-v2/layer_0",
        ],
        help="Path to load the probe from.",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        nargs="+",
        default=[
            "probes/pythia-70m-aclimdb-v2/best/eval-aclimdb",
            "probes/pythia-70m-aclimdb-v2/layer_0/eval-aclimdb",
        ],
        help="Path to save the eval results to.",
    )
    parser.add_argument(
        "--layer_id",
        type=int,
        nargs="*",
        default=None,
        help="Hidden layer ID(s) to extract features from.",
    )

    args = parser.parse_args()
    main(args)
