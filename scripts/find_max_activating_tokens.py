import json
from typing import Any, Dict

import numpy as np
import safetensors
import safetensors.numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm as tq
from tqdm.auto import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast


def get_closest(input: torch.Tensor, embeddings: torch.Tensor):
    """Given points in an embedding space and a dictionary of valid points, returns the
    closest valid points, distances to the closest points, and indices of the closest
    points."""

    num_tokens = input.shape[1]

    input = input  # (batch_size, num_tokens, embedding_dim)
    embeddings = embeddings  # (vocab_size, embedding_dim)

    # Compute indices of closest tokens without gradients
    closest_idx = []
    with torch.no_grad():
        for token_id in range(num_tokens):
            distances = input[None, :, token_id] - embeddings[:, None]  # (v, b, e)
            distances = torch.linalg.vector_norm(distances, dim=-1)  # (v, b)

            sorted_idx = torch.argsort(distances, dim=0, descending=False)
            closest_idx.append(sorted_idx[0])

    closest_idx = torch.stack(closest_idx, dim=1)

    # Compute closest embeddings and distances to previously identified tokens with
    # gradients
    closest_embeddings = embeddings[closest_idx]
    closest_distance = input - closest_embeddings
    closest_distance = torch.linalg.vector_norm(closest_distance, dim=-1)

    return closest_embeddings, closest_distance, closest_idx


class StraightThroughEstimator(torch.autograd.Function):
    """Returns the first argument during the forward pass. Routes gradients to the
    second argument during the backward pass."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, replacement: torch.Tensor):
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return None, grad_output


def exists(x: Any | None) -> bool:
    return x is not None


def main():
    # Parameters
    top_k = 10000
    num_tokens = 16
    layer_id = 18
    target_label = 1.0
    device = "cuda"

    prefix = None
    postfix = None

    target_text = "Hello, how are you?"

    # Directories
    model_path = "EleutherAI/pythia-410m"
    probe_path = "probes/pythia-410m-aclimdb"
    dictionary_path = "dictionaries/aclimdb"

    # Load model, tokenizer and dataset
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device, torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.to_bettertransformer()
    # model.gradient_checkpointing_enable()
    model.eval()
    model.requires_grad_(False)

    # Print info on model, tokenizer and dataset
    print("---MODEL---")
    print(model)
    print("---TOKENIZER---")
    print(tokenizer)

    # Get embeddings
    embedding_layer: nn.Embedding = model.get_input_embeddings()
    embedding_matrix = embedding_layer.weight.data.clone()
    embedding_id_to_token_id_mapping = None

    # Prepare pre- and postfix ids and embeddings
    if exists(prefix):
        prefix_token_ids = (
            tokenizer(prefix, add_special_tokens=False, return_tensors="pt")
            .to(device)
            .input_ids
        )
        prefix_token_embeddings = embedding_matrix[prefix_token_ids]

    if exists(postfix):
        postfix_token_ids = (
            tokenizer(postfix, add_special_tokens=False, return_tensors="pt")
            .to(device)
            .input_ids
        )
        postfix_token_embeddings = embedding_matrix[postfix_token_ids]

    # Construct target
    if exists(target_text):
        target_token_ids = (
            tokenizer(target_text, add_special_tokens=False, return_tensors="pt")
            .to(device)
            .input_ids
        )
        target_token_embeddings = embedding_matrix[target_token_ids]

    # If necessary, limit to dictionary top-k tokens
    if dictionary_path is not None:
        with open(f"{dictionary_path}/counts_ids.json", mode="r") as f:
            token_id_counts: Dict[str, int] = json.load(f)

        topk_token_ids = [int(x) for x in token_id_counts.keys()][:top_k]
        embedding_matrix = embedding_matrix[topk_token_ids].clone()

        embedding_id_to_token_id_mapping = {i: v for i, v in enumerate(topk_token_ids)}

    vocab_size, embedding_dim = embedding_matrix.shape

    # Initialize embeddings from random tokens
    init_token_ids = torch.randint(vocab_size, size=(1, num_tokens))
    init_token_ids = init_token_ids.to(device)

    input_token_embeddings = embedding_matrix[init_token_ids].clone()
    input_token_embeddings = input_token_embeddings.to(
        dtype=torch.float32, device=device
    )
    input_token_embeddings.requires_grad_(True)

    # Construct probe
    if exists(probe_path):
        probe_params = safetensors.numpy.load_file(f"{probe_path}/probe.safetensors")
        probe = nn.Linear(embedding_dim, 1)

        with torch.no_grad():
            probe.weight.data.copy_(torch.from_numpy(probe_params["weight"]))
            probe.bias.data.copy_(torch.from_numpy(probe_params["bias"]))

        probe.to(dtype=torch.float32, device=device)
        probe.requires_grad_(False)

        target_label = torch.FloatTensor([[target_label]]).to(device)

    # Construct optimizer
    optimizer = optim.Adam([input_token_embeddings], lr=1e-3, betas=(0.0, 0.99))

    # Optimize embeddings
    pbar = trange(4096)
    for i in pbar:
        # Quantize embeddings to dictionary
        closest_embeddings, closest_distances, _ = get_closest(
            input_token_embeddings, embedding_matrix
        )

        closest_embeddings: torch.Tensor = StraightThroughEstimator.apply(
            closest_embeddings, input_token_embeddings
        )

        # Construct input embeddings
        to_merge = []
        if exists(prefix):
            to_merge.append(prefix_token_embeddings)
        to_merge.append(closest_embeddings)
        if exists(postfix):
            to_merge.append(postfix_token_embeddings)
        if exists(target_text):
            to_merge.append(target_token_embeddings)

        merged_embeddings = torch.cat(to_merge, dim=1)

        # Get model outputs
        outputs: CausalLMOutputWithPast = model(
            inputs_embeds=merged_embeddings.to(model.dtype), output_hidden_states=True
        )

        # Calculate losses
        reg_loss = torch.mean(closest_distances**2)
        # reg_loss = torch.mean(closest_distances)

        probe_loss = 0.0
        if exists(probe_path):
            features = outputs.hidden_states[layer_id][:, -1]
            logits = probe(features.to(torch.float32))

            # probe_loss = F.binary_cross_entropy_with_logits(logits, target_label)
            probe_loss = -torch.mean(logits * target_label)

        target_loss = 0.0
        if exists(target_text):
            logits = outputs.logits[0, -target_token_ids.shape[1] - 1 : -1]
            target_loss = F.cross_entropy(logits, target_token_ids[0])

        loss = probe_loss + target_loss  # + reg_loss * 1e1

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_postfix_str(
            f"Loss: {loss:.2e} Probe: {probe_loss:.2e} Target {target_loss:.2e} Reg: {reg_loss:.2e}"
        )

    # Convert back to token ids and token string
    max_token_ids = get_closest(input_token_embeddings, embedding_matrix)[2]
    max_token_ids = max_token_ids.cpu().numpy().tolist()[0]

    if embedding_id_to_token_id_mapping is not None:
        max_token_ids = [embedding_id_to_token_id_mapping[v] for v in max_token_ids]

    max_tokens = tokenizer.decode(max_token_ids)
    print(max_tokens)

    to_merge = []
    if exists(prefix):
        to_merge.append(prefix_token_ids)
    to_merge.append(torch.LongTensor([max_token_ids]).to(device))
    if exists(postfix):
        to_merge.append(postfix_token_ids)

    merged_token_ids = torch.cat(to_merge, dim=1)
    merged_token_ids_list = merged_token_ids.cpu().numpy().tolist()[0]
    merged_tokens = tokenizer.decode(merged_token_ids_list)
    print("---")
    print(merged_tokens)

    # Validate
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device, torch_dtype=torch.float32
    )

    reconstructed_token_inputs = tokenizer(merged_tokens, return_tensors="pt").to(
        device
    )

    if exists(probe_path):
        # Load probe
        probe_params = safetensors.numpy.load_file(f"{probe_path}/probe.safetensors")
        probe = nn.Linear(embedding_dim, 1)

        with torch.no_grad():
            probe.weight.data.copy_(torch.from_numpy(probe_params["weight"]))
            probe.bias.data.copy_(torch.from_numpy(probe_params["bias"]))
            probe.to(dtype=torch.float32, device=device)

        # Get probe loss
        with torch.no_grad():
            outputs: CausalLMOutputWithPast = model(
                **reconstructed_token_inputs,
                output_hidden_states=True,
            )

            features = outputs.hidden_states[layer_id][:, -1]
            logits = probe(features.to(torch.float32))

            # probe_loss = F.binary_cross_entropy_with_logits(logits, target_label)
            probe_loss = -torch.mean(logits * target_label)
            print(probe_loss)

    if exists(target_text):
        with torch.no_grad():
            generated = model.generate(
                **reconstructed_token_inputs, max_new_tokens=len(target_text)
            )
            print(tokenizer.batch_decode(generated)[0])


if __name__ == "__main__":
    main()
