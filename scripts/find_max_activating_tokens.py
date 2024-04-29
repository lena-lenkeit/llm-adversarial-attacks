import pathlib

import datasets
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


class EmbeddingEstimator(torch.autograd.Function):
    """Given points in an embedding space and a dictionary of valid points, snaps the
    points to the closest valid points during the forward pass, but acts like a
    straight-through estimator on the backward pass, similar to a VQ-VAE."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, embeddings: torch.Tensor):
        return get_closest(input, embeddings)[0]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def main():
    # Parameters
    top_k = 100000000000
    num_tokens = 4
    layer_id = 18
    device = "cuda"

    # Directories
    model_path = "EleutherAI/pythia-410m"
    probe_path = "probes/pythia-410m-aclimdb"

    # Load model, tokenizer and dataset
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device, torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.to_bettertransformer()
    model.eval()
    model.requires_grad_(False)

    # Print info on model, tokenizer and dataset
    print("---MODEL---")
    print(model)
    print("---TOKENIZER---")
    print(tokenizer)

    # Initialize embeddings from random tokens
    input_embedding_layer: nn.Embedding = model.get_input_embeddings()
    vocab_size, embedding_dim = input_embedding_layer.weight.shape

    input_token_ids = torch.randint(min(vocab_size, top_k), size=(1, num_tokens))
    input_token_ids = input_token_ids.to(device)

    input_embeddings = input_embedding_layer(input_token_ids)
    input_embeddings.requires_grad_(True)

    print(input_embeddings.shape)

    closest_embeddings = EmbeddingEstimator.apply(
        input_embeddings, input_embedding_layer.weight[:top_k]
    )

    print(input_embeddings)
    print(closest_embeddings)

    # Construct probe
    probe_params = safetensors.numpy.load_file(f"{probe_path}/probe.safetensors")
    probe = nn.Linear(embedding_dim, 1)
    probe = probe.to(device)

    probe.requires_grad_(False)

    with torch.no_grad():
        probe.weight.data.copy_(torch.from_numpy(probe_params["weight"]))
        probe.bias.data.copy_(torch.from_numpy(probe_params["bias"]))

    # Optimize embeddings
    optimizer = optim.Adam([input_embeddings], lr=1e-2)
    target_label = torch.FloatTensor([[1.0]]).to(device)

    pbar = trange(128)
    for i in pbar:
        closest_embeddings = EmbeddingEstimator.apply(
            input_embeddings, input_embedding_layer.weight[:top_k]
        )

        # closest_embeddings = input_embeddings

        outputs: CausalLMOutputWithPast = model(
            inputs_embeds=closest_embeddings, output_hidden_states=True
        )

        features = outputs.hidden_states[layer_id][:, -1]
        logits = probe(features)

        probe_loss = F.binary_cross_entropy_with_logits(logits, target_label)
        reg_loss = torch.mean(
            get_closest(input_embeddings, input_embedding_layer.weight[:top_k])[1]
        )

        loss = probe_loss + reg_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_postfix_str(
            f"Loss: {loss:.2e} Probe: {probe_loss:.2e} Reg: {reg_loss:.2e}"
        )

    max_token_ids = get_closest(input_embeddings, input_embedding_layer.weight[:top_k])[
        2
    ]

    print(max_token_ids)
    print(tokenizer.decode(max_token_ids[0]))

    # Validate
    with torch.no_grad():
        outputs: CausalLMOutputWithPast = model(
            input_ids=max_token_ids, output_hidden_states=True
        )

        features = outputs.hidden_states[layer_id][:, -1]
        logits = probe(features)

        probe_loss = F.binary_cross_entropy_with_logits(logits, target_label)
        print(probe_loss)


if __name__ == "__main__":
    main()
