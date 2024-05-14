import argparse
import json
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import safetensors
import safetensors.numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import einsum, rearrange
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


class RerouteGradients(torch.autograd.Function):
    """Returns the first argument during the forward pass. Routes gradients to the
    second argument during the backward pass."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, replacement: torch.Tensor):
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return None, grad_output


def reroute_gradients(input: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
    return RerouteGradients.apply(input, replacement)


def exists(x: Any | None) -> bool:
    return x is not None


class ConstantOrSchedule:
    def __init__(self, values: List[float] | None, num_steps: int):
        print(values)

        if values is None:
            self.schedule = None
        elif len(values) == 1:
            self.schedule = np.full(num_steps, values[0], dtype=np.float32)
        elif len(values) == 2:
            self.schedule = np.geomspace(
                values[0], values[1], num=num_steps, dtype=np.float32
            )

    def __getitem__(self, index: int):
        if self.schedule is None:
            raise ValueError("Schedule used, but not initialized properly!")

        return self.schedule[index]


def main(args: argparse.Namespace):
    # Infer values
    has_prefix = exists(args.prefix_text)
    has_postfix = exists(args.postfix_text)
    has_probe = exists(args.probe_path)
    has_target = exists(args.target_text)
    has_dictionary = exists(args.dictionary_path)

    # TODO: Auto probe layer id from metadata if not supplied by user

    # Load model, tokenizer and dataset
    if args.dtype == "auto":
        model_dtype = "auto"
    else:
        model_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=args.device,
        torch_dtype=model_dtype,
        attn_implementation=args.attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.to_bettertransformer:
        model = model.to_bettertransformer()
    if args.gradient_checkpointing:
        model = model.gradient_checkpointing_enable()

    model.eval()
    model.requires_grad_(False)

    # Print info on model, tokenizer and dataset
    if args.verbose:
        print("---MODEL---")
        print(model)
        print("---TOKENIZER---")
        print(tokenizer)

    # Get embeddings (and cast to float32 to avoid precision issues when training with
    # lower-precision models)
    embedding_layer: nn.Embedding = model.get_input_embeddings()
    embedding_matrix = embedding_layer.weight.data.clone().to(
        dtype=torch.float32, device=args.device
    )
    embedding_id_to_token_id_mapping = None

    def tokenize_text(text: str) -> torch.LongTensor:
        return (
            tokenizer(text, add_special_tokens=False, return_tensors="pt")
            .to(args.device)
            .input_ids
        )

    def embedd_text(text: str):
        token_ids = tokenize_text(text)
        return embedding_matrix[token_ids], token_ids

    # Prepare pre- and postfix ids and embeddings
    if has_prefix:
        prefix_token_embeddings, prefix_token_ids = embedd_text(args.prefix_text)
    if has_postfix:
        postfix_token_embeddings, postfix_token_ids = embedd_text(args.postfix_text)

    # Construct target
    if has_target:
        target_token_embeddings, target_token_ids = embedd_text(args.target_text)

    # If necessary, limit to dictionary top-k tokens
    if has_dictionary:
        with open(f"{args.dictionary_path}/counts_ids.json", mode="r") as f:
            token_id_counts: Dict[str, int] = json.load(f)

        topk_token_ids = [int(x) for x in token_id_counts.keys()]
        if exists(args.dictionary_top_k):
            topk_token_ids = topk_token_ids[: args.dictionary_top_k]

        embedding_matrix = embedding_matrix[topk_token_ids].clone()
        embedding_id_to_token_id_mapping = {i: v for i, v in enumerate(topk_token_ids)}

    vocab_size, embedding_dim = embedding_matrix.shape

    # Initialize embeddings from random tokens
    if args.space == "embeddings":
        init_token_ids = torch.randint(vocab_size, size=(1, args.num_tokens))
        init_token_ids = init_token_ids.to(args.device)

        input_token_embeddings = embedding_matrix[init_token_ids].clone()
        input_token_embeddings.requires_grad_(True)

    # Construct probe (again in float32 to avoid precision issues)
    if has_probe:
        probe_params = safetensors.numpy.load_file(
            f"{args.probe_path}/probe.safetensors"
        )
        probe = nn.Linear(embedding_dim, 1)

        with torch.no_grad():
            probe.weight.data.copy_(torch.from_numpy(probe_params["weight"]))
            probe.bias.data.copy_(torch.from_numpy(probe_params["bias"]))

        probe.to(dtype=torch.float32, device=args.device)
        probe.requires_grad_(False)

        target_label = torch.FloatTensor([[args.probe_target_label]]).to(args.device)

    # Construct optimizer
    # TODO: Add handling for adam betas and general optim kwargs
    optim_cls = {"sgd": optim.SGD, "adam": optim.Adam}[args.optim_cls]

    if args.space == "tokens":
        token_mixing_logits = torch.randn(
            (1, args.num_tokens, vocab_size),
            device=args.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        with torch.no_grad():
            token_mixing_logits.data = token_mixing_logits.data * 0.0

        optimizer = optim_cls([token_mixing_logits], lr=0.0)
    elif args.space == "embeddings":
        optimizer = optim_cls([input_token_embeddings], lr=0.0)

    lr_schedule = ConstantOrSchedule(args.lr, args.num_steps)
    reg_schedule = ConstantOrSchedule(args.regularization, args.num_steps)
    temp_schedule = ConstantOrSchedule(args.temperature, args.num_steps)
    tau_schedule = ConstantOrSchedule(args.tau, args.num_steps)

    best_loss = torch.inf
    if args.space == "tokens":
        best_token_values = token_mixing_logits.clone()
    elif args.space == "embeddings":
        best_token_values = input_token_embeddings.clone()

    # Optimize embeddings
    pbar = trange(args.num_steps)
    for i in pbar:
        if args.space == "embeddings":
            # Quantize embeddings to dictionary
            closest_embeddings, closest_sq_distances, closest_idx, sq_dist = (
                get_closest(input_token_embeddings, embedding_matrix)
            )

            if args.quantize_embeddings:
                quantized_embeddings = reroute_gradients(
                    closest_embeddings, input_token_embeddings
                )
            else:
                quantized_embeddings = input_token_embeddings

            if args.projection_constraint:
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
                quantized_embeddings = torch.where(
                    project_mask, projected_embeddings, input_token_embeddings
                )
        elif args.space == "tokens":
            if args.method == "soft_softmax":
                token_mixing_soft = F.softmax(
                    token_mixing_logits * temp_schedule[i], dim=-1
                )

                soft_embeddings = einsum(
                    token_mixing_soft,
                    embedding_matrix,
                    "... vocab, vocab features -> ... features",
                )

                quantized_embeddings = soft_embeddings
            elif args.method == "hard_softmax":
                token_mixing_soft = F.softmax(
                    token_mixing_logits * temp_schedule[i], dim=-1
                )
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

                quantized_embeddings = reroute_gradients(
                    hard_embeddings, soft_embeddings
                )
            elif args.method == "hard_gumbel_softmax":
                token_mixing_factors = F.gumbel_softmax(
                    token_mixing_logits * temp_schedule[i],
                    tau=tau_schedule[i],
                    hard=True,
                )
                quantized_embeddings = einsum(
                    token_mixing_factors,
                    embedding_matrix,
                    "... vocab, vocab features -> ... features",
                )

        # Construct input embeddings
        to_merge = []
        if has_prefix:
            to_merge.append(prefix_token_embeddings)
        to_merge.append(quantized_embeddings)
        if has_postfix:
            to_merge.append(postfix_token_embeddings)
        if has_target:
            to_merge.append(target_token_embeddings)

        merged_embeddings = torch.cat(to_merge, dim=1)

        # Get model outputs
        outputs: CausalLMOutputWithPast = model(
            inputs_embeds=merged_embeddings.to(model.dtype), output_hidden_states=True
        )

        # Calculate losses
        reg_loss = 0.0
        if args.regularization_type == "entropy":
            token_mixing_probs = token_mixing_soft
            token_mixing_logprobs = F.log_softmax(
                token_mixing_logits * temp_schedule[i], dim=-1
            )
            entropy = -torch.sum(token_mixing_probs * token_mixing_logprobs, dim=-1)
            reg_loss = torch.mean(entropy)
        elif args.regularization_type == "max_probability":
            token_mixing_probs = token_mixing_soft
            reg_loss = -torch.mean(torch.max(token_mixing_probs, dim=-1)[0])
        elif args.regularization_type == "abs_dist":
            reg_loss = torch.mean(torch.sqrt(torch.clip(sq_dist, min=1e-10)))
        elif args.regularization_type == "closest_sq_dist":
            reg_loss = torch.mean(closest_sq_distances)
        elif args.regularization_type == "closest_log_sq_dist":
            reg_loss = torch.mean(
                torch.log(torch.clip(closest_sq_distances, min=1e-10))
            )
        elif args.regularization_type == "softmin_kernel":
            reg_loss = torch.mean(softmin_kernel(sq_dist, temp_schedule[i]))
        elif args.regularization_type == "hard_kernel":
            reg_loss = torch.mean(hard_kernel(sq_dist, temp_schedule[i]))

        probe_loss = 0.0
        if has_probe:
            features = outputs.hidden_states[args.layer_id][:, -1]
            logits = probe(features.to(torch.float32))

            if args.probe_loss == "bce":
                probe_loss = F.binary_cross_entropy_with_logits(logits, target_label)
            if args.probe_loss == "direct":
                probe_loss = -torch.mean(logits * target_label)

        target_loss = 0.0
        if has_target:
            logits = outputs.logits[0, -target_token_ids.shape[1] - 1 : -1]
            target_loss = F.cross_entropy(logits, target_token_ids[0])

        realism_loss = 0.0
        if args.realism_loss:
            start = 0
            if has_prefix:
                start = prefix_token_embeddings.shape[0]

            offset = closest_idx.shape[1]
            end = start + offset

            realism_loss = F.cross_entropy(
                outputs.logits[0, start : end - 1],
                closest_idx[0, start + 1 : end],
            )

        loss = probe_loss + target_loss + reg_loss * reg_schedule[i] + realism_loss

        # Store best tokens
        if loss < best_loss:
            best_loss = loss
            if args.space == "tokens":
                best_token_values = token_mixing_logits.clone()
            elif args.space == "embeddings":
                best_token_values = input_token_embeddings.clone()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if not args.per_token_line_search:
            optimizer.step()

        if args.space == "embeddings":
            # Line Search
            if args.per_token_line_search:
                with torch.no_grad():
                    origin = quantized_embeddings
                    direction = input_token_embeddings.grad
                    points = embedding_matrix

                    direction = direction / torch.linalg.vector_norm(
                        direction, dim=-1, keepdim=True
                    )

                    for token_id in range(args.num_tokens):
                        _origin = origin[0, token_id]
                        _direction = direction[0, token_id]

                        offset = rearrange(
                            points, "vocab dim -> vocab dim"
                        ) - rearrange(_origin, "dim -> 1 dim")

                        dotproduct = einsum(
                            offset * rearrange(_direction, "dim -> 1 dim"),
                            "vocab dim -> vocab",
                        )

                        projected = (
                            rearrange(dotproduct, "vocab -> vocab 1") * _direction
                        )

                        sq_dist = torch.linalg.vector_norm(offset - projected, dim=-1)
                        sq_dist = torch.where(dotproduct > 0, torch.inf, sq_dist)
                        sq_dist[closest_idx[0]] = torch.inf
                        input_token_embeddings.data[0, token_id] = points[
                            torch.argmin(sq_dist)
                        ]

            # PGD
            if args.project_embeddings:
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

        # Apply stages
        if args.num_stages > 1:
            stage_length = args.num_steps // args.num_stages
            current_stage = i // stage_length

            # Set correct lr
            optimizer.param_groups[0]["lr"] = (
                lr_schedule[i] * args.stage_lr_multiplier**current_stage
            )

            # Reset values to best found so far when passing a stage
            if ((i + 1) % stage_length) == 0:
                with torch.no_grad():
                    if args.space == "tokens":
                        token_mixing_logits.data = best_token_values.clone()
                    elif args.space == "embeddings":
                        input_token_embeddings.data = best_token_values.clone()

        # Update lr
        if not args.num_stages > 1:
            optimizer.param_groups[0]["lr"] = lr_schedule[i]

        # Add postfix to pbar
        postfix_dict = {
            "best_total_loss": best_loss,
            "total_loss": loss,
            "reg_loss": reg_loss,
            "likelihood_loss": realism_loss,
        }
        if args.space == "embeddings":
            postfix_dict["mean_sq_dist"] = torch.mean(closest_sq_distances)
        if has_probe:
            postfix_dict["probe_loss"] = probe_loss
        if has_target:
            postfix_dict["target_loss"] = target_loss

        postfix_str = " ".join([f"{k}: {v:.2e}" for k, v in postfix_dict.items()])
        pbar.set_postfix_str(postfix_str)

    # Convert back to token ids and token string
    if args.space == "tokens":
        # Ids of tokens with largest logits
        max_token_ids = torch.argmax(best_token_values, dim=2)
    elif args.space == "embeddings":
        # Ids of closest tokens in dictionary
        max_token_ids = get_closest(best_token_values, embedding_matrix)[2]

    max_token_ids_list = max_token_ids.cpu().numpy().tolist()[0]

    if embedding_id_to_token_id_mapping is not None:
        max_token_ids_list = [
            embedding_id_to_token_id_mapping[v] for v in max_token_ids_list
        ]

    max_tokens = tokenizer.decode(max_token_ids_list)
    print("---Found Tokens---")
    print(max_tokens)
    print("------------------")

    to_merge = []
    if has_prefix:
        to_merge.append(prefix_token_ids)
    to_merge.append(max_token_ids)
    if has_postfix:
        to_merge.append(postfix_token_ids)

    merged_token_ids = torch.cat(to_merge, dim=1)
    merged_token_ids_list = merged_token_ids.cpu().numpy().tolist()[0]
    merged_tokens = tokenizer.decode(merged_token_ids_list)
    print("---Merged Tokens---")
    print(merged_tokens)
    print("-------------------")

    # Validate
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=args.device,
        torch_dtype=model_dtype,
        attn_implementation=args.attn_implementation,
    )
    if args.to_bettertransformer:
        model = model.to_bettertransformer()
    if args.gradient_checkpointing:
        model = model.gradient_checkpointing_enable()

    model.eval()
    model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    reconstructed_token_inputs = tokenizer(
        merged_tokens, add_special_tokens=False, return_tensors="pt"
    ).to(args.device)

    if has_probe:
        # Load probe
        probe_params = safetensors.numpy.load_file(
            f"{args.probe_path}/probe.safetensors"
        )
        probe = nn.Linear(embedding_dim, 1)

        with torch.no_grad():
            probe.weight.data.copy_(torch.from_numpy(probe_params["weight"]))
            probe.bias.data.copy_(torch.from_numpy(probe_params["bias"]))
            probe.to(dtype=torch.float32, device=args.device)

        # Get probe loss
        with torch.no_grad():
            outputs: CausalLMOutputWithPast = model(
                **reconstructed_token_inputs,
                output_hidden_states=True,
            )

            features = outputs.hidden_states[args.layer_id][:, -1]
            logits = probe(features.to(torch.float32))

            if args.probe_loss == "bce":
                probe_loss = F.binary_cross_entropy_with_logits(logits, target_label)
            if args.probe_loss == "direct":
                probe_loss = -torch.mean(logits * target_label)
            print(probe_loss)

    if has_target:
        with torch.no_grad():
            generated = model.generate(
                **reconstructed_token_inputs,
                max_new_tokens=target_token_ids.shape[1] * 4,
            )
            print(tokenizer.batch_decode(generated)[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Finds the tokens minimizing a configurable loss (e.g. the label
        predicted by a hidden activation linear probe, likelihood of continuing with a
        specific text, likelihood of the found tokens, and more.)"""
    )

    parser.add_argument(
        "--num_tokens",
        type=int,
        default=16,
        help="Numbers of adversarial tokens to train",
    )
    parser.add_argument(
        "--layer_id",
        type=int,
        default=None,
        help="Hidden layer id at which the probe is applied",
    )
    parser.add_argument(
        "--probe_target_label",
        type=float,
        default=None,
        help="Target label or logit for the probe",
    )
    parser.add_argument(
        "--prefix_text", type=str, default=None, help="Prefix text to be used"
    )
    parser.add_argument(
        "--postfix_text", type=str, default=None, help="Postfix text to be used"
    )
    parser.add_argument("--target_text", type=str, default=None, help="Target text")
    parser.add_argument(
        "--probe_path", type=str, default=None, help="Path to the probe file"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to the model"
    )
    parser.add_argument(
        "--dictionary_path", type=str, default=None, help="Path to the dictionary file"
    )
    parser.add_argument(
        "--dictionary_top_k",
        type=int,
        default=None,
        help="Number of top-k tokens (by frequency) to include from the dictionary",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Data type for model parameters",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Attention implementation to use",
    )
    parser.add_argument(
        "--to_bettertransformer",
        action="store_true",
        help="Whether to use bettertransformer",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed model and tokenizer info, and found tokens",
    )

    parser.add_argument(
        "--optim_cls",
        type=str,
        choices=["sgd", "adam"],
        default="sgd",
        help="Optimizer class to use",
    )
    parser.add_argument(
        "--space",
        type=str,
        choices=["tokens", "embeddings"],
        default="embeddings",
        help="Optimization space",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["soft_softmax", "hard_softmax", "hard_gumbel_softmax"],
        default="soft_softmax",
        help="Optimization method when optimizing in token space",
    )
    parser.add_argument(
        "--lr",
        type=float,
        nargs="*",
        default=[1e-3],
        help="Learning rate or learning rate schedule",
    )
    parser.add_argument(
        "--num_steps", type=int, default=1024, help="Number of optimization steps"
    )
    parser.add_argument(
        "--regularization",
        type=float,
        nargs="*",
        default=[1.0],
        help="Regularization coefficient or schedule",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        nargs="*",
        default=[1.0],
        help="Temperature for softmax or similar techniques",
    )
    parser.add_argument(
        "--tau",
        type=float,
        nargs="*",
        default=[1.0],
        help="Tau for Gumbel softmax or similar techniques",
    )
    parser.add_argument(
        "--num_stages", type=int, default=4, help="Number of stages for optimization"
    )
    parser.add_argument(
        "--stage_lr_multiplier",
        type=float,
        default=0.1,
        help="Learning rate multiplier for each stage",
    )
    parser.add_argument(
        "--regularization_type",
        type=str,
        default=None,
        help="Type of regularization to apply",
    )
    parser.add_argument(
        "--probe_loss",
        type=str,
        choices=["bce", "direct"],
        default="direct",
        help="Loss type for probe",
    )
    parser.add_argument(
        "--realism_loss", action="store_true", help="Apply realism loss"
    )
    parser.add_argument(
        "--quantize_embeddings", action="store_true", help="Quantize embeddings"
    )
    parser.add_argument(
        "--projection_constraint",
        action="store_true",
        help="Apply projection constraint",
    )
    parser.add_argument(
        "--per_token_line_search",
        action="store_true",
        help="Enable per token line search",
    )
    parser.add_argument(
        "--project_embeddings", action="store_true", help="Project embeddings"
    )

    args = parser.parse_args()
    main(args)
