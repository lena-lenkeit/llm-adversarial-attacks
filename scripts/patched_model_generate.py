"""Sample continuations / generations from a model patched along a probe."""

import argparse
from functools import partial
from typing import Any, Tuple

import safetensors
import safetensors.numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast


def exists(x: Any | None) -> bool:
    return x is not None


def load_model_and_tokenizer(
    model_path: str,
    device: str,
    dtype: str,
    attn_implementation: str,
    to_bettertransformer: bool | None,
    cache_dir: str | None,
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
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if to_bettertransformer:
        model = model.to_bettertransformer()

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


def residual_patch_fn(
    residual: torch.Tensor, patch: torch.Tensor, patch_all: bool, additive: bool
):
    if not patch_all and residual.shape[1] == 1:
        return residual

    if additive:
        patched = residual + patch
    else:
        patched = torch.broadcast_to(patch, residual.shape)

    if not patch_all:
        return torch.cat((residual[:, :-1], patched[:, -1:]), dim=1)
    else:
        return patched


def patched_layer_hook(
    module: nn.Module,
    args: Any,
    output: Tuple[torch.Tensor, ...],
    patch_fn,
) -> Tuple[torch.Tensor, ...]:
    patched = patch_fn(output[0])
    return (patched,) + output[1:]


def patched_output_hook(
    module: nn.Module,
    input: Tuple[torch.Tensor],
    patch_fn,
) -> Tuple[torch.Tensor]:
    patched = patch_fn(input[0])
    return (patched,)


@torch.inference_mode()
def main(args: argparse.Namespace):
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.device,
        args.dtype,
        args.attn_implementation,
        args.to_bettertransformer,
        args.model_cache_dir,
    )

    # Load probe
    probe, probe_layer_id, _ = load_probe(
        args.probe_path, model.config.hidden_size, args.layer_id, args.device
    )

    # Create intervention
    intervention = rearrange(probe.weight[0], "d -> 1 1 d") * args.strength
    print(intervention)

    # Patch forward hook into model
    num_model_layers = model.config.num_hidden_layers
    is_model_output = probe_layer_id == num_model_layers

    patch_fn = partial(
        residual_patch_fn,
        patch=intervention,
        patch_all=args.patch_all,
        additive=args.additive,
    )

    if isinstance(model, GPTNeoXForCausalLM):
        if is_model_output:
            model.gpt_neox.final_layer_norm.register_forward_pre_hook(
                partial(patched_output_hook, patch_fn=patch_fn)
            )
        else:
            model.gpt_neox.layers[probe_layer_id].register_forward_hook(
                partial(patched_layer_hook, patch_fn=patch_fn)
            )

    if isinstance(model, LlamaForCausalLM):
        if is_model_output:
            model.model.norm.register_forward_pre_hook(
                partial(patched_output_hook, patch_fn=patch_fn)
            )
        else:
            model.model.layers[probe_layer_id].register_forward_hook(
                partial(patched_layer_hook, patch_fn=patch_fn)
            )

    tokens = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    outputs = model.generate(**tokens, max_new_tokens=128)
    print(tokenizer.batch_decode(outputs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Sample from a language model while patching one of the
                    activations as defined by a linear probe."""
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="I just watched a movie in the cinema, and here's my honest review:",
        help="Pre-prompt to use.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="EleutherAI/pythia-410m",
        help="Path to the model",
    )
    parser.add_argument(
        "--model_cache_dir", type=str, help="Directory to cache downloaded models to."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "float32", "float16", "bfloat16"],
        default="float32",
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
        "--probe_path",
        type=str,
        default="probes/pythia-410m-aclimdb-v2/best_test",
        help="Path to the probe file",
    )
    parser.add_argument(
        "--layer_id",
        type=int,
        default=None,
        help="Hidden layer id at which the probe is applied",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.1,
        help="Strength of patch, as a multiplier of the probe features.",
    )
    parser.add_argument(
        "--patch_all",
        action="store_true",
        help="Set to patch at all positions in the sequence.",
    )
    parser.add_argument(
        "--additive", action="store_true", help="Set to add patch to residuals."
    )

    args = parser.parse_args()
    main(args)
