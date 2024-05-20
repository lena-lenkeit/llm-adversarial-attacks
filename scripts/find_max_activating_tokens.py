import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Literal, Tuple

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


def safetensors_load_file_metadata(filename: str):
    with open(filename, mode="rb") as f:
        # Read header length (uint64)
        header_length = int.from_bytes(f.read(8), byteorder="little", signed=False)

        # Read header (utf-8 encoded json)
        header_dict = json.loads(f.read(header_length).decode("utf-8"))

    return header_dict["__metadata__"]


def get_cross_entropy_metrics(
    logits: torch.FloatTensor,
    targets: torch.LongTensor,
    dim: int | Tuple[int, ...] | None = None,
):
    cross_entropy = F.cross_entropy(logits, targets, reduction="none")
    log_probs = -cross_entropy

    probs = torch.exp(log_probs)
    total_log_prob = log_probs.sum(dim=dim)
    mean_log_prob = torch.log(probs.mean(dim=dim))

    total_prob = torch.exp(total_log_prob)
    mean_prob = torch.exp(mean_log_prob)

    return {
        "log_probs": log_probs,
        "probs": probs,
        "total_log_prob": total_log_prob,
        "mean_log_prob": mean_log_prob,
        "total_prob": total_prob,
        "mean_prob": mean_prob,
    }


def torch_dict_to_python(torch_dict: Dict[str, torch.Tensor]):
    return {k: v.tolist() for k, v in torch_dict.items()}


def dict_prefix_keys(dictionary: dict, prefix: str):
    return {prefix + k: v for k, v in dictionary.items()}


def realism_loss_input_helper(
    logits: torch.FloatTensor,
    adv_token_ids: torch.LongTensor,
    has_prefix: bool,
    num_prefix_tokens: int,
    num_adv_tokens: int,
):
    start = 0
    offset = num_adv_tokens

    if has_prefix:
        start = num_prefix_tokens - 1
        offset += 1

    end = start + offset

    adv_logits = rearrange(logits[:, start : end - 1], "b t f -> b f t")
    adv_targets = rearrange(adv_token_ids[:, 1:], "b t -> b t")

    return adv_logits, adv_targets


def probe_loss_fn(
    probe: nn.Module,
    probe_layer_id: int,
    hidden_activations: Tuple[torch.FloatTensor, ...],
    target: torch.FloatTensor,
    num_input_tokens: int,
    loss_type: Literal["bce", "direct"],
    logit_bias: float = 0.0,
    logit_scale: float = 1.0,
):
    features = hidden_activations[probe_layer_id][:, num_input_tokens - 1]
    logits = probe(features.to(torch.float32))
    z = (logits + logit_bias) * logit_scale

    if loss_type == "bce":
        probe_loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        probe_loss = torch.mean(probe_loss, dim=1)
    elif loss_type == "direct":
        probe_loss = -torch.mean(logits * target, dim=1)
    elif loss_type == "z_direct":
        probe_loss = -torch.mean(z * target, dim=1)

    return probe_loss, logits, z


def target_loss_input_helper(
    logits: torch.FloatTensor, target_token_ids: torch.LongTensor, num_input_tokens: int
):
    return rearrange(logits[:, num_input_tokens:], "b t f -> b f t"), rearrange(
        target_token_ids, "b t -> b t"
    )


def target_loss_fn(logits: torch.FloatTensor, targets: torch.LongTensor):
    return F.cross_entropy(logits, targets)


def load_model_and_tokenizer(
    model_path: str,
    device: str,
    dtype: str,
    attn_implementation: str,
    to_bettertransformer: bool | None,
    gradient_checkpointing: bool | None,
):
    if dtype == "auto":
        model_dtype = "auto"
    else:
        model_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=model_dtype,
        attn_implementation=attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if to_bettertransformer:
        model = model.to_bettertransformer()
    if gradient_checkpointing:
        model = model.gradient_checkpointing_enable()

    model.eval()
    model.requires_grad_(False)

    return model, tokenizer


@torch.no_grad()
def load_probe(
    probe_path: str, embedding_dim: int, probe_layer_id: int | None, device: str
):
    # Load probe data
    probe_data = safetensors.numpy.load_file(f"{probe_path}/probe.safetensors")
    probe_eval = safetensors.numpy.load_file(f"{probe_path}/eval.safetensors")

    # Initialize probe
    probe = nn.Linear(embedding_dim, 1)
    probe.weight.data.copy_(torch.from_numpy(probe_data["weight"]))
    probe.bias.data.copy_(torch.from_numpy(probe_data["bias"]))

    probe.to(dtype=torch.float32, device=device)
    probe.requires_grad_(False)

    # Load correct probe layer
    if not exists(probe_layer_id):
        probe_layer_id = int(probe_data["train_layer_id"])

    # Infer scale and bias for z loss
    bias = -probe_eval["test_logits"].mean()
    scale = 1 / probe_eval["test_logits"].std()

    return probe, probe_layer_id, (bias, scale)


def mix_embeddings(mixing_factors: torch.Tensor, embedding_matrix: torch.Tensor):
    mixed_embeddings = einsum(
        mixing_factors,
        embedding_matrix,
        "... vocab, vocab features -> ... features",
    )

    return mixed_embeddings


def check_roundtrip(input: torch.LongTensor, roundtrip: torch.LongTensor):
    # Check shapes
    shape_check = input.shape == roundtrip.shape

    # Check contents
    content_check = False
    if shape_check:
        content_check = torch.all(input == roundtrip)

    return shape_check and content_check


def batch_check_roundtrip(input: torch.LongTensor, roundtrip: torch.LongTensor):
    return torch.BoolTensor([check_roundtrip(i, r) for i, r in zip(input, roundtrip)])


def main(args: argparse.Namespace):
    # TODO: Multi-probe, Multi-Target optimization
    # TODO: Investigate the case where the dec-enc roundtrip fails
    # TODO: Add special tokens for correctness, if they exist
    # TODO: Check over this entire script again, clean up stuff, make sure impl. is good
    # TODO: Think about what to do with step-optimization and token space

    # NOTE: Letting it train in embedding space with lr 1e2 for a long time seems to do
    # wonders for the realism loss. With 4096 steps and 4 1e-1 stages, I got the realism
    # loss down to 2.79 . Maybe training even longer, without stages, a different lr
    # would give super realistic samples.

    # Infer values
    has_prefix = exists(args.prefix_text)
    has_postfix = exists(args.postfix_text)
    has_probe = exists(args.probe_path)
    has_target = exists(args.target_text)
    has_dictionary = exists(args.dictionary_path)
    probe_layer_id = args.layer_id

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.device,
        args.dtype,
        args.attn_implementation,
        args.to_bettertransformer,
        args.gradient_checkpointing,
    )

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
    num_adv_tokens = args.num_tokens
    num_input_tokens = num_adv_tokens
    num_prefix_tokens = None
    num_postfix_tokens = None
    num_target_tokens = None

    if has_prefix:
        prefix_token_embeddings, prefix_token_ids = embedd_text(args.prefix_text)
        num_prefix_tokens = prefix_token_ids.shape[1]
        num_input_tokens += num_prefix_tokens
    if has_postfix:
        postfix_token_embeddings, postfix_token_ids = embedd_text(args.postfix_text)
        num_postfix_tokens = postfix_token_ids.shape[1]
        num_input_tokens += num_postfix_tokens

    # Construct target
    num_total_tokens = num_input_tokens

    if has_target:
        target_token_embeddings, target_token_ids = embedd_text(args.target_text)
        num_target_tokens = target_token_ids.shape[1]
        num_total_tokens += num_target_tokens

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
        init_token_ids = torch.randint(
            vocab_size, size=(args.batch_size, args.num_tokens)
        )
        init_token_ids = init_token_ids.to(args.device)

        input_token_embeddings = embedding_matrix[init_token_ids].clone()
        input_token_embeddings.requires_grad_(True)

    # Construct probe (again in float32 to avoid precision issues)
    probe_layer_id = args.layer_id
    if has_probe:
        probe, probe_layer_id, (probe_logit_bias, probe_logit_scale) = load_probe(
            args.probe_path, embedding_dim, probe_layer_id, args.device
        )
        target_label = torch.FloatTensor(
            [[args.probe_target_label]] * args.batch_size
        ).to(args.device)

    # Construct optimizer
    # TODO: Add handling for adam betas and general optim kwargs
    optim_cls = {"sgd": optim.SGD, "adam": optim.Adam}[args.optim_cls]

    if args.space == "tokens":
        token_mixing_logits = torch.randn(
            (args.batch_size, args.num_tokens, vocab_size),
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

                soft_embeddings = mix_embeddings(token_mixing_soft, embedding_matrix)

                quantized_embeddings = soft_embeddings
            elif args.method == "hard_softmax":
                token_mixing_soft = F.softmax(
                    token_mixing_logits * temp_schedule[i], dim=-1
                )
                token_mixing_hard = F.softmax(token_mixing_logits * 1e10, dim=-1)

                soft_embeddings = mix_embeddings(token_mixing_soft, embedding_matrix)
                hard_embeddings = mix_embeddings(token_mixing_hard, embedding_matrix)

                quantized_embeddings = reroute_gradients(
                    hard_embeddings, soft_embeddings
                )
            elif args.method == "hard_gumbel_softmax":
                token_mixing_factors = F.gumbel_softmax(
                    token_mixing_logits * temp_schedule[i],
                    tau=tau_schedule[i],
                    hard=True,
                )

                quantized_embeddings = mix_embeddings(
                    token_mixing_factors, embedding_matrix
                )

            _, _, closest_idx, _ = get_closest(quantized_embeddings, embedding_matrix)

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
            reg_loss = torch.mean(entropy, dim=1)
        elif args.regularization_type == "max_probability":
            token_mixing_probs = token_mixing_soft
            reg_loss = -torch.mean(torch.max(token_mixing_probs, dim=-1)[0], dim=1)
        elif args.regularization_type == "abs_dist":
            reg_loss = torch.mean(torch.sqrt(torch.clip(sq_dist, min=1e-10)), dim=1)
        elif args.regularization_type == "closest_sq_dist":
            reg_loss = torch.mean(closest_sq_distances, dim=1)
        elif args.regularization_type == "closest_log_sq_dist":
            reg_loss = torch.mean(
                torch.log(torch.clip(closest_sq_distances, min=1e-10)), dim=1
            )
        elif args.regularization_type == "softmin_kernel":
            reg_loss = torch.mean(softmin_kernel(sq_dist, temp_schedule[i]), dim=1)
        elif args.regularization_type == "hard_kernel":
            reg_loss = torch.mean(hard_kernel(sq_dist, temp_schedule[i]), dim=1)

        probe_loss = 0.0
        if has_probe:
            probe_loss, probe_logits, probe_z = probe_loss_fn(
                probe,
                probe_layer_id,
                outputs.hidden_states,
                target_label,
                num_input_tokens,
                args.probe_loss,
                probe_logit_bias,
                probe_logit_scale,
            )

        target_loss = 0.0
        if has_target:
            logits, targets = target_loss_input_helper(
                outputs.logits, target_token_ids, num_input_tokens
            )
            target_loss = F.cross_entropy(logits, targets, reduction="none")
            target_loss = torch.mean(target_loss, dim=1)

        realism_loss = 0.0
        if args.realism_loss:
            logits, targets = realism_loss_input_helper(
                outputs.logits,
                closest_idx,
                has_prefix,
                num_prefix_tokens,
                num_adv_tokens,
            )
            realism_loss = F.cross_entropy(logits, targets, reduction="none")
            realism_loss = torch.mean(realism_loss, dim=1)

        loss = probe_loss + target_loss + reg_loss * reg_schedule[i] + realism_loss

        # Store best tokens
        with torch.no_grad():
            update_best_mask = loss < best_loss
            best_loss = torch.where(update_best_mask, loss, best_loss)

            if args.space == "tokens":
                best_token_values = torch.where(
                    rearrange(update_best_mask, "b -> b 1 1"),
                    token_mixing_logits,
                    best_token_values,
                )
            elif args.space == "embeddings":
                best_token_values = torch.where(
                    rearrange(update_best_mask, "b -> b 1 1"),
                    closest_embeddings,
                    best_token_values,
                )

        loss = torch.mean(loss, dim=0)

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
            "lr": optimizer.param_groups[0]["lr"],
            "best_loss": best_loss,
            "loss": loss,
        }

        if args.num_stages > 1:
            postfix_dict["stage"] = current_stage
        if args.space == "embeddings":
            postfix_dict["mean_sq_dist"] = torch.mean(closest_sq_distances)
        if has_probe:
            postfix_dict["probe_loss"] = probe_loss
            postfix_dict["probe_logits"] = probe_logits
            postfix_dict["probe_z"] = probe_z
        if has_target:
            postfix_dict["target_loss"] = target_loss
        if args.regularization_type:
            postfix_dict["reg_loss"] = reg_loss
        if args.realism_loss:
            postfix_dict["ll_loss"] = realism_loss

        def to_python(x):
            if isinstance(x, torch.Tensor):
                x = x.mean().item()

            if isinstance(x, float):
                x = f"{x:.2e}"
            elif isinstance(x, int):
                x = f"{x}"
            else:
                raise TypeError(x, type(x))

            return x

        postfix_str = " ".join(
            [f"{k}: {to_python(v)}" for k, v in postfix_dict.items()]
        )
        pbar.set_postfix_str(postfix_str)

    # Get best token ids
    if args.space == "tokens":
        # Ids of tokens with largest logits
        adv_token_ids = torch.argmax(best_token_values, dim=2)
    elif args.space == "embeddings":
        # Ids of closest tokens in dictionary
        adv_token_ids = get_closest(best_token_values, embedding_matrix)[2]

    # Backtranslate if dictionary was used
    if has_dictionary:
        mapping = torch.LongTensor(embedding_id_to_token_id_mapping.values())
        mapping = mapping.to(args.device)

        adv_token_ids = mapping[adv_token_ids]

    # Reconstruct full input (including prefix and postfix)
    to_merge = []
    if has_prefix:
        to_merge.append(prefix_token_ids)
    to_merge.append(adv_token_ids)
    if has_postfix:
        to_merge.append(postfix_token_ids)

    input_token_ids = torch.cat(to_merge, dim=1)

    if has_target:
        to_merge.append(target_token_ids)

    full_token_ids = torch.cat(to_merge, dim=1)

    # Decode all to strings
    adv_tokens = tokenizer.batch_decode(adv_token_ids)
    input_tokens = tokenizer.batch_decode(input_token_ids)
    full_tokens = tokenizer.batch_decode(full_token_ids)

    # Re-encode for round-trip consistency check
    def tokenize_text(text: str):
        return tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=args.num_tokens,
            return_tensors="pt",
        ).input_ids.to(args.device)

    roundtrip_adv_token_ids = tokenize_text(adv_tokens)
    roundtrip_input_token_ids = tokenize_text(input_tokens)
    roundtrip_full_token_ids = tokenize_text(full_tokens)

    # Check if round-trip decoding-encoding recovers the input
    roundtrip_adv_check = batch_check_roundtrip(adv_token_ids, roundtrip_adv_token_ids)
    roundtrip_input_check = batch_check_roundtrip(
        input_token_ids, roundtrip_input_token_ids
    )
    roundtrip_full_check = batch_check_roundtrip(
        full_token_ids, roundtrip_full_token_ids
    )

    roundtrip_all_check = torch.logical_and(
        torch.logical_and(roundtrip_adv_check, roundtrip_input_check),
        roundtrip_full_check,
    )
    if not torch.all(roundtrip_all_check):
        print("At least one batch failed the roundtrip check! Check output manually...")
        print(roundtrip_all_check)

    # Reload model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.device,
        args.dtype,
        args.attn_implementation,
        args.to_bettertransformer,
        args.gradient_checkpointing,
    )

    with torch.no_grad():
        # Get logits and hidden states for all tokens
        full_outputs: CausalLMOutputWithPast = model(
            full_token_ids, output_hidden_states=True
        )

        # Realism loss
        logits, targets = realism_loss_input_helper(
            full_outputs.logits,
            adv_token_ids,
            has_prefix,
            num_prefix_tokens,
            num_adv_tokens,
        )
        realism_metrics = get_cross_entropy_metrics(logits, targets, dim=1)

    if has_probe:
        # Load probe
        probe, _, (probe_logit_bias, probe_logit_scale) = load_probe(
            args.probe_path, embedding_dim, probe_layer_id, args.device
        )

        # Get probe loss
        with torch.no_grad():
            probe_loss, probe_logits, probe_z = probe_loss_fn(
                probe,
                probe_layer_id,
                full_outputs.hidden_states,
                target_label,
                num_input_tokens,
                args.probe_loss,
                probe_logit_bias,
                probe_logit_scale,
            )

            print(f"Probe Loss : {probe_loss.mean().item():.2e}")

    if has_target:
        with torch.no_grad():
            # Get target loss
            logits, targets = target_loss_input_helper(
                full_outputs.logits, target_token_ids, num_input_tokens
            )
            target_metrics = get_cross_entropy_metrics(logits, targets, dim=1)

            # Sample most likely continuation
            generated_token_ids = model.generate(
                full_token_ids, max_new_tokens=target_token_ids.shape[1] * 4
            )
            generated_tokens = tokenizer.batch_decode(generated_token_ids)

            print(f"Model Continuation: {generated_tokens}")

    # Save results
    if exists(args.output_path):
        os.makedirs(f"{args.output_path}", exist_ok=True)
        with open(f"{args.output_path}/adversarial_tokens.json", mode="w") as f:
            data = {
                "roundtrip_adv_check": roundtrip_adv_check.tolist(),
                "roundtrip_input_check": roundtrip_input_check.tolist(),
                "roundtrip_full_check": roundtrip_full_check.tolist(),
                "roundtrip_all_check": roundtrip_all_check.tolist(),
                "adv_token_ids": adv_token_ids.tolist(),
                "adv_tokens": adv_tokens,
            }

            data.update(
                dict_prefix_keys(torch_dict_to_python(realism_metrics), "realism_")
            )

            if has_probe:
                data["probe_logits"] = probe_logits.tolist()
                data["probe_z"] = probe_z.tolist()
                data["probe_targets"] = target_label.tolist()

            if has_target:
                data.update(
                    dict_prefix_keys(torch_dict_to_python(target_metrics), "target_")
                )

                data["generated_token_ids"] = generated_token_ids.tolist()
                data["generated_tokens"] = generated_tokens

            json.dump(data, f, indent=4)


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
        "--output_path", type=str, default=None, help="Path at which to store results"
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
        default="hard_softmax",
        help="Optimization method when optimizing in token space",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of adversarial sequences to find in parallel",
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
        choices=["bce", "direct", "z_direct"],
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
