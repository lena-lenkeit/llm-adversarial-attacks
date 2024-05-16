import argparse
from typing import Any, Dict, List

import torch
from einops import rearrange
from tqdm.auto import tqdm as tq
from tqdm.auto import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

import datasets


@torch.inference_mode()
def main(args: argparse.Namespace):
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
    dataset = datasets.load_from_disk(args.dataset_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    if args.to_bettertransformer:
        model = model.to_bettertransformer()
    model.eval()

    # Limit dataset length
    for split in dataset:
        if len(dataset[split]) <= args.max_elements:
            continue

        indices = range(args.max_elements)
        dataset[split] = dataset[split].shuffle(args.seed, keep_in_memory=True)
        dataset[split] = dataset[split].flatten_indices(keep_in_memory=True)
        dataset[split] = dataset[split].select(indices, keep_in_memory=True)

    # Tokenize dataset
    def tokenize_fn(rows: Dict[str, Any]):
        text: List[str] = rows["text"]
        encoding = tokenizer(text, max_length=args.max_length, truncation=True)
        token_ids = encoding.input_ids
        num_tokens = [len(ids) for ids in token_ids]

        return {"token_ids": token_ids, "num_tokens": num_tokens}

    dataset = dataset.map(tokenize_fn, batched=True, keep_in_memory=True)

    # Sort by input lengths to speed up batch encoding
    dataset = dataset.sort("num_tokens", reverse=True, keep_in_memory=True)
    dataset = dataset.flatten_indices(keep_in_memory=True)

    # Print info on model, tokenizer and dataset
    if args.verbose:
        print("---MODEL---")
        print(model)
        print("---TOKENIZER---")
        print(tokenizer)
        print("---DATASET---")
        print(dataset)

    # Run entire dataset through model, caching all hidden layer activations at the last
    # token
    for split in dataset:
        activation_cache = {}
        for offset in trange(0, len(dataset[split]), args.batch_size):
            # Get batch of texts
            rows = dataset[split][offset : offset + args.batch_size]
            token_ids: List[int] = rows["token_ids"]

            # Pad tokens
            tokens = tokenizer.pad(
                {"input_ids": token_ids},
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(args.device)

            # Get last token indices
            last_token_idx = torch.argmin(tokens.attention_mask, dim=1) - 1
            last_token_idx = last_token_idx % tokens.attention_mask.shape[1]
            last_token_idx = rearrange(last_token_idx, "i -> i 1 1")

            # Get LM outputs
            outputs: CausalLMOutputWithPast = model(**tokens, output_hidden_states=True)

            # Save hidden layer activations to cache
            for cache_id, hidden_state in enumerate(outputs.hidden_states):
                # Get last token activations
                activations = torch.take_along_dim(hidden_state, last_token_idx, dim=1)
                activations = activations[:, 0]
                activations = activations.cpu().to(torch.float32).numpy().tolist()

                # Save to cache
                cache = activation_cache.get(cache_id, [])
                cache.extend(activations)
                activation_cache[cache_id] = cache

        # Append cache to dataset
        for cache_id in activation_cache:
            dataset[split] = dataset[split].add_column(
                f"hidden_{cache_id}", activation_cache[cache_id]
            )

    # Save dataset, with activations included
    dataset.save_to_disk(args.activation_cache_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Appends hidden layer activations to a dataset."
    )
    parser.add_argument(
        "--max_elements",
        type=int,
        default=4096,
        help="Maximum number of elements in the dataset to process.",
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum length for tokenization."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for encoding."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="dtype to load the model in.",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Which attention implementation to use. Support varies between models.",
    )
    parser.add_argument(
        "-b",
        "--to_bettertransformer",
        action="store_true",
        help="Set to use bettertransformers. Support varies between models.",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Seed to use for shuffling the dataset."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Set to print additional output.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="EleutherAI/pythia-160m",
        help="Path to the pretrained model.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/aclimdb",
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--activation_cache_path",
        type=str,
        default="activations/pythia-160m-aclimdb",
        help="Path to save the cached activations to.",
    )

    args = parser.parse_args()
    main(args)
