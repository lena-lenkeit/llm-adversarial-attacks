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

    input = input[None]  # (1, batch_size, num_tokens, embedding_dim)
    embeddings = embeddings[:, None]  # (vocab_size, 1, embedding_dim)

    new = []
    for token_id in range(num_tokens):
        distances = input[:, :, token_id] - embeddings
        distances = torch.linalg.vector_norm(distances, dim=-1)

        closest_idx = torch.argsort(distances, dim=0, descending=False)[0]
        new.append(embeddings[closest_idx])

    return torch.cat(new, dim=1)


class EmbeddingEstimator(torch.autograd.Function):
    """Given points in an embedding space and a dictionary of valid points, snaps the
    points to the closest valid points during the forward pass, but acts like a
    straight-through estimator on the backward pass, similar to a VQ-VAE."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, embeddings: torch.Tensor):
        return get_closest(input, embeddings)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def main():
    # Parameters
    num_tokens = 4
    layer_id = 6
    device = "cuda"

    # Directories
    model_path = "EleutherAI/pythia-70m"
    probe_path = "probes/pythia-70m-aclimdb"

    # Load model, tokenizer and dataset
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device, torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

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

    input_token_ids = torch.randint(vocab_size, size=(1, num_tokens))
    input_token_ids = input_token_ids.to(device)

    input_embeddings = input_embedding_layer(input_token_ids)
    input_embeddings.requires_grad_(True)

    print(input_embeddings.shape)

    closest_embeddings = EmbeddingEstimator.apply(
        input_embeddings, input_embedding_layer.weight
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
            input_embeddings, input_embedding_layer.weight
        )

        # closest_embeddings = input_embeddings

        outputs: CausalLMOutputWithPast = model(
            inputs_embeds=closest_embeddings, output_hidden_states=True
        )

        features = outputs.hidden_states[layer_id][:, -1]
        logits = probe(features)

        loss = F.binary_cross_entropy_with_logits(logits, target_label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_postfix_str(f"Loss: {loss:.2e}")


if __name__ == "__main__":
    main()
