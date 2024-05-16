# Adversarial Attacks on LLMs and LLM Activation Probes

This is a work-in-progress repository for finding adversarial strings of tokens to influence Large Language Models (LLMs) in a variety of ways, as part of investigating generalization and robustness of LLM activation probes.

## Features

Currently, supported adversarial optimization targets are:
- Forcing linear probes on top of LLM hidden layer activations to have a certain score.
- Forcing certain continuations of the prompt.
- Optionally concatenating the adversarial prompt with a prefix and/or postfix string.

Additionally, the adversarial prompt can be optimized for naturalness (high likelihood).

Supported optimization methods are:
- Optimization in token space (before the embedding layer; as (soft) one-hot vectors)
  - By passing the token with the highest value during the forward pass, but routing gradients back to a softmax in the backward pass (works).
  - Via a hard Gumbel softmax (works).
  - With a softmax and mixing between token embeddings plus a regularization term to approximate a one-hot vector (doesn't work well). 
- Optimization in embedding space (after the embedding layer; as points in embedding space)
  - By clamping embeddings to the dictionary during the forward pass, but routing gradients to the unclamped embeddings during the backward pass (works).
  - By directly optimizing unclamped embeddings, but adding a regularization term to guide embeddings towards points in the dictionary (difficult to get working).
    - Includes a variety of regularization terms and kernels.
  - By constraining embeddings to be close to nearby embeddings in the dictionary, with increasing constraint tightness over time (doesn't work well).
  - Similar to above, but with projected gradient descent (doesn't work well).
  - Optionally using per-token line search instead of SGD (doesn't work well, but implementation also doesn't consider the ideal criterion).

All methods perform continuous optimization via Stochastic Gradient Descent (SGD), but outputs are hard token IDs/strings, via the tricks above.

## Utilities

Included utilities allow:
- Populating a text dataset with last-token activations at all hidden layers.
- Training linear probes on top of hidden layer activations for classification tasks (e.g., sentiment analysis).
- Finding discrete adversarial prompts (token IDs/strings) to influence probe classification scores and/or generated continuations.

## Observations

Some observations:
- Finding adversarial inputs for probes works incredibly well, even for large models (I tested it up to Llama 3 7B so far).
- Finding adversarial inputs to steer generations is much harder, but possible. For short target strings, I managed to exactly force the target to be generated with greedy sampling.
- Naturalness is difficult to optimize; the log-likelihood of the text typically stays pretty low.
- Optimized adversarial prompts typically are semi-garbled text. They aren't fully coherent, even when optimizing for naturalness, but seem somewhat interpretable. When forcing outputs, they also aren't an exact copy of the target text, though they tend to include tokens of the target text.

## Relevant Papers

- Universal and Transferable Adversarial Attacks on Aligned Language Models (https://arxiv.org/abs/2307.15043)
  - Optimization Method: Greedy Coordinate Gradient-based Search (Gradients w.r.t. one-hot token vectors, find top-k gradient subset, find replacement tokens across top-k subset with the lowest loss when used as a replacement, repeat).
  - Finds prompt suffixes for transferable adversarial attacks.
  - Outputs are semi-garbage.
- Hard Prompts Made Easy: Gradient-Based Discrete Optimization for Prompt Tuning and Discovery (https://arxiv.org/pdf/2302.03668)
  - Optimization Method: Continuous optimization in embedding space, projection to nearest-neighbors in the vocabulary during forward pass, gradient applied to unprojected embeddings in the backward pass.
  - They also use a fluency loss to make the found text more interpretable.

## To-Do

- Finish converting everything over to argparse
- Add multi-probe and multi-target optimization support
- Add artifact IO to all scripts
- Run a benchmark on all datasets across all Pythia sizes
- Add plotting scripts for results
- Write blog post on results