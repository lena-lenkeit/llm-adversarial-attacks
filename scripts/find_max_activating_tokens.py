import json
from typing import Any, Dict

import numpy as np
import safetensors
import safetensors.numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import einsum, rearrange
from torch.distributions import RelaxedOneHotCategorical
from tqdm.auto import tqdm as tq
from tqdm.auto import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast


def get_closest(input: torch.Tensor, embeddings: torch.Tensor):
    """Given points in an embedding space and a dictionary of valid points, returns the
    closest valid points, squared distances to the closest points, and indices of the
    closest points."""

    sq_dist = (
        rearrange(einsum(input**2, "... dim -> ..."), "... seq -> ... seq 1")
        + rearrange(einsum(embeddings**2, "... dim -> ..."), "vocab -> 1 vocab")
        - 2 * einsum(input, embeddings, "... seq dim, vocab dim -> ... seq vocab")
    )

    closest_sq_distances, closest_idx = torch.min(sq_dist, dim=-1)
    closest_embeddings = embeddings[closest_idx]

    return closest_embeddings, closest_sq_distances, closest_idx, sq_dist


def softmin_kernel(sq_distances: torch.Tensor, temperature: float = 1.0):
    softmin_distances = F.softmin(sq_distances * (1 / temperature), dim=-1)

    return torch.sum(softmin_distances * sq_distances, dim=-1)


def hard_kernel(sq_distances, dist_border: float = 1.0):
    return torch.clamp(torch.min(sq_distances, dim=1)[0] / dist_border**2, min=1.0)


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
    layer_id = 6
    target_label = 1.0
    device = "cuda"

    prefix = None
    postfix = None

    target_text = "Nice to meet you!"

    # Directories
    model_path = "EleutherAI/pythia-410m"
    probe_path = None  # "probes/pythia-70m-aclimdb"
    dictionary_path = None

    # Load model, tokenizer and dataset
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device, torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.to_bettertransformer()
    # model.gradient_checkpointing_enable()
    # model.eval()
    model.train()
    model.requires_grad_(False)

    # Print info on model, tokenizer and dataset
    print("---MODEL---")
    print(model)
    print("---TOKENIZER---")
    print(tokenizer)

    # Get embeddings
    embedding_layer: nn.Embedding = model.get_input_embeddings()
    embedding_matrix = embedding_layer.weight.data.clone().to(
        dtype=torch.float32, device=device
    )
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
    # """
    token_mixing_logits = torch.randn(
        (1, num_tokens, vocab_size),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    with torch.no_grad():
        token_mixing_logits.data = token_mixing_logits.data * 0.0

    # optimizer = optim.Adam([token_mixing_logits], lr=1e-0, betas=(0.9, 0.99))
    optimizer = optim.SGD([token_mixing_logits], lr=1e1)
    # """

    # optimizer = optim.Adam([input_token_embeddings], lr=1e-4, betas=(0.0, 0.99))
    # optimizer = optim.SGD([input_token_embeddings], lr=1e2)

    num_steps = 1024 * 4
    lr_schedule = np.geomspace(1e2, 1e2, num_steps)
    reg_schedule = np.geomspace(1e0, 1e0, num_steps)
    temp_schedule = np.geomspace(1e0, 1e0, num_steps)
    tau_schedule = np.geomspace(1e1, 1e-1, num_steps)

    num_stages = 8
    best_loss = torch.inf
    best_token_ids = init_token_ids.clone()

    # Optimize embeddings
    pbar = trange(num_steps)
    for i in pbar:
        # Quantize embeddings to dictionary
        # closest_embeddings, closest_sq_distances, closest_idx, sq_dist = get_closest(
        #    input_token_embeddings, embedding_matrix
        # )

        # quantized_embeddings: torch.Tensor = StraightThroughEstimator.apply(
        #    closest_embeddings, input_token_embeddings
        # )

        # quantized_embeddings = input_token_embeddings

        # print(token_mixing_logits)
        token_mixing_soft = F.softmax(token_mixing_logits * temp_schedule[i], dim=-1)
        token_mixing_hard = F.softmax(token_mixing_logits * 1e10, dim=-1)

        soft_embeddings = einsum(
            token_mixing_soft,
            embedding_matrix,
            "... vocab, vocab features -> ... features",
        )

        hard_embeddings = einsum(
            token_mixing_hard,
            embedding_matrix,
            "... vocab, vocab features -> ... features",
        )

        quantized_embeddings: torch.Tensor = StraightThroughEstimator.apply(
            hard_embeddings, soft_embeddings
        )

        """
        token_mixing_factors = F.gumbel_softmax(
            -torch.log(torch.clip(sq_dist, min=1e-10)) * temp_schedule[i],
            tau=tau_schedule[i],
            hard=True,
        )
        print(token_mixing_factors)
        quantized_embeddings = einsum(
            token_mixing_factors,
            embedding_matrix,
            "... vocab, vocab features -> ... features",
        )
        """

        """
        with torch.no_grad():
            if (i + 1) % 64 == 0:
                closest_embeddings, closest_sq_distances, _, sq_dist = get_closest(
                    input_token_embeddings, embedding_matrix
                )

                input_token_embeddings.data = closest_embeddings.data.clone()

        quantized_embeddings = input_token_embeddings
        """

        """
        print(token_mixing_logits.max(dim=2))
        # token_mixing_probs = F.softmax(token_mixing_logits * temp_schedule[i], dim=-1)
        token_mixing_factors = F.gumbel_softmax(
            token_mixing_logits * temp_schedule[i], tau=tau_schedule[i], hard=True
        )
        quantized_embeddings = einsum(
            token_mixing_factors,
            embedding_matrix,
            "... vocab, vocab features -> ... features",
        )
        """

        """
        distance_limit = temp_schedule[i]
        closest_distances = torch.sqrt(
            torch.clip(closest_sq_distances[..., None], min=1e-8)
        )

        normalized_offsets = (
            input_token_embeddings - closest_embeddings
        ) / closest_distances
        projected_embeddings = normalized_offsets * distance_limit + closest_embeddings

        project_mask = closest_distances > distance_limit
        quantized_embeddings = torch.where(
            project_mask, projected_embeddings, input_token_embeddings
        )
        """

        # Construct input embeddings
        to_merge = []
        if exists(prefix):
            to_merge.append(prefix_token_embeddings)
        to_merge.append(quantized_embeddings)
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
        reg_loss = 0.0
        """
        token_mixing_logprobs = F.log_softmax(
            token_mixing_logits * temp_schedule[i], dim=-1
        )
        entropy = -torch.sum(token_mixing_probs * token_mixing_logprobs, dim=-1)
        reg_loss = torch.mean(entropy)
        """
        # reg_loss = -torch.mean(torch.max(token_mixing_probs, dim=2)[0])
        # reg_loss = torch.mean(torch.sqrt(torch.clip(sq_dist, min=1e-10)))
        # reg_loss = torch.mean(closest_sq_distances)
        # reg_loss = torch.mean(torch.log(torch.clip(closest_sq_distances, min=1e-10)))
        # reg_loss = torch.mean(softmin_kernel(sq_dist, temp_schedule[i]))
        # reg_loss = torch.mean(hard_kernel(sq_dist, temp_schedule[i]))

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

        realism_loss = 0.0
        # realism_loss = F.cross_entropy(
        #    outputs.logits[0, : closest_idx.shape[1] - 1], closest_idx[0, 1:]
        # )

        loss = probe_loss + target_loss + reg_loss * reg_schedule[i] + realism_loss

        # if loss < best_loss:
        #    best_loss = loss
        #    best_token_ids = closest_idx

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Line search
        """
        with torch.no_grad():
            origin = quantized_embeddings
            direction = input_token_embeddings.grad
            points = embedding_matrix

            direction = direction / torch.linalg.vector_norm(
                direction, dim=-1, keepdim=True
            )

            for token_id in range(num_tokens):
                _origin = origin[0, token_id]
                _direction = direction[0, token_id]

                offset = rearrange(points, "vocab dim -> vocab dim") - rearrange(
                    _origin, "dim -> 1 dim"
                )

                dotproduct = einsum(
                    offset * rearrange(_direction, "dim -> 1 dim"),
                    "vocab dim -> vocab",
                )

                projected = rearrange(dotproduct, "vocab -> vocab 1") * _direction

                sq_dist = torch.linalg.vector_norm(offset - projected, dim=-1)
                sq_dist = torch.where(dotproduct > 0, torch.inf, sq_dist)
                sq_dist[closest_idx[0]] = torch.inf
                input_token_embeddings.data[0, token_id] = points[torch.argmin(sq_dist)]
        """

        # PGD
        """
        with torch.no_grad():
            distance_limit = temp_schedule[i]
            closest_distances = torch.sqrt(
                torch.clip(closest_sq_distances[..., None], min=1e-8)
            )

            normalized_offsets = (
                input_token_embeddings - closest_embeddings
            ) / closest_distances
            projected_embeddings = (
                normalized_offsets * distance_limit + closest_embeddings
            )

            project_mask = closest_distances > distance_limit
            input_token_embeddings.data = torch.where(
                project_mask, projected_embeddings, input_token_embeddings
            )
        """

        # optimizer.param_groups[0]["lr"] = lr_schedule[i]
        # if ((i + 1) % (num_steps // num_stages)) == 0:
        #    with torch.no_grad():
        #        input_token_embeddings.data = embedding_matrix[best_token_ids]
        #        optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] / 10
        # SqDist: {torch.mean(closest_sq_distances):.2e}
        pbar.set_postfix_str(
            f"Loss: {loss:.2e} Probe: {probe_loss:.2e} Target {target_loss:.2e} Reg: {reg_loss:.2e}  Realism: {realism_loss:.2e}"
        )

    with torch.no_grad():
        input_token_embeddings.data = embedding_matrix[best_token_ids]

    # Convert back to token ids and token string
    # max_token_ids = get_closest(input_token_embeddings, embedding_matrix)[2]
    max_token_ids = torch.max(token_mixing_logits, dim=2)[1]
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
    # model = AutoModelForCausalLM.from_pretrained(
    #    model_path, device_map=device, torch_dtype=torch.float32
    # )

    reconstructed_token_inputs = tokenizer(
        merged_tokens, add_special_tokens=False, return_tensors="pt"
    ).to(device)

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
                **reconstructed_token_inputs,
                max_new_tokens=target_token_ids.shape[1] * 4,
            )
            print(tokenizer.batch_decode(generated)[0])


if __name__ == "__main__":
    main()
