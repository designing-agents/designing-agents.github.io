"""
Reward Models

Post-training starts with people comparing two replies to the same prompt and
picking the better one. A reward model learns to predict those picks, and a
model is then trained to score higher on it. You will label some pairs
yourself and then train a reward model on 1,000 pairs in which a GPT-4 reply
is marked better than a Llama-2 reply.

    pip install fastembed torch huggingface_hub numpy
    python reward_model.py
"""

from __future__ import annotations

import hashlib
import json
import textwrap
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
warnings.filterwarnings("ignore", message=".*encountered in matmul")

HERE = Path(__file__).resolve().parent
DATASET = "Intel/orca_dpo_pairs"
DATA_FILE = "orca_rlhf.jsonl"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_CACHE = HERE / ".reward_model_embeddings.npz"

N_TRAIN = 1000
N_TEST = 200
N_LABEL = 8
LR = 0.01
EPOCHS = 200

_embedder = None


# =============================================================================
# Step 1: Setup (written for you)
# =============================================================================
# Downloads the dataset and the embedding model, and embeds every reply. The
# first run takes a few minutes. Later runs read everything from disk.
#
#   dataset: https://huggingface.co/datasets/Intel/orca_dpo_pairs

def download() -> list[dict]:
    """Every row of the Orca DPO pairs file, as a dict."""
    path = hf_hub_download(DATASET, DATA_FILE, repo_type="dataset")
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_pairs(rows: list[dict]) -> list[dict]:
    """
    In:  rows from download()
    Out: one pair per row, a dict with keys "prompt", "chosen", and "rejected"

    "chosen" is GPT-4's reply and "rejected" is Llama-2's.
    """
    return [{"prompt": r["question"], "chosen": r["chosen"],
             "rejected": r["rejected"]} for r in rows]


def embed(texts: list[str]) -> np.ndarray:
    """As in HW2 Task 3.1. One row per text, saved to disk."""
    global _embedder
    key = hashlib.sha256("\0".join(texts).encode()).hexdigest()[:16]
    cache = {}
    if EMBED_CACHE.exists():
        with np.load(EMBED_CACHE) as saved:
            cache = dict(saved)
    if key not in cache:
        from fastembed import TextEmbedding
        if _embedder is None:
            _embedder = TextEmbedding(EMBED_MODEL)
        print(f"  embedding {len(texts)} replies")
        cache[key] = np.array(list(_embedder.embed(texts)))
        np.savez(EMBED_CACHE, **cache)
    return cache[key]


def words(text: str) -> int:
    return len(text.split())


def setup() -> tuple[list[dict], list[dict], dict]:
    """
    Out: (train, test, X), where X["train"] and X["test"] each hold the
         embeddings of the chosen and the rejected replies, row i from pair i
    """
    rows = download()
    train = load_pairs(rows[:N_TRAIN])
    test = load_pairs(rows[N_TRAIN:N_TRAIN + N_TEST])
    X = {name: (embed([p["chosen"] for p in pairs]),
                embed([p["rejected"] for p in pairs]))
         for name, pairs in (("train", train), ("test", test))}
    print(f"setup done: {len(train)} train pairs, {len(test)} test pairs")
    return train, test, X


# =============================================================================
# Step 2: Annotation
# =============================================================================

# --- Task 1: Be the annotator ------------------------------------------------
# Run the file. It prints eight prompts, each with two replies. For each one,
# decide which reply is better and record 1 or 2, in the order printed. The
# file then shows which reply the dataset marks as better.

MY_LABELS: list[int] = []   # TODO


# =============================================================================
# Step 3: Reward function
# =============================================================================

# --- Task 2: A baseline ------------------------------------------------------
# Before training anything, check how far a trivial rule gets: always prefer
# the longer reply.

def length_baseline(pairs: list[dict]) -> float:
    """
    In:  pairs from load_pairs
    Out: the share of pairs in which the chosen reply has more words than the
         rejected one, where a tie counts as half right
    """
    # TODO
    return 0.0


# --- Task 3: Train ------------------------------------------------------------
# The reward model gives each reply a score: its embedding times a weight
# vector w. The Bradley-Terry model turns the difference between two scores
# into the probability that the first reply is preferred, by passing it
# through a sigmoid. Training maximizes the probability of every preference in
# the data. Let torch compute the gradient.
#
#   the loss:  https://arxiv.org/abs/2203.02155 (section 3.5)
#   autograd:  https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
#   optimizer: https://pytorch.org/docs/stable/optim.html

def fit_reward(X_chosen: np.ndarray, X_rejected: np.ndarray, lr: float = LR,
               epochs: int = EPOCHS) -> tuple[np.ndarray, list[float]]:
    """
    In:  embeddings of the chosen and rejected replies, both (n, d), row i of
         each from the same pair, a step size, and how many steps to take
    Out: (w, losses), where w is a numpy array of shape (d,) and losses holds
         the training loss after each step

    Start w at zeros and use every pair at every step.
    """
    # TODO
    return np.zeros(X_chosen.shape[1]), []


# --- Task 4: Evaluate --------------------------------------------------------

def accuracy(w: np.ndarray, X_chosen: np.ndarray,
             X_rejected: np.ndarray) -> float:
    """
    In:  a weight vector and embeddings of chosen and rejected replies
    Out: the share of pairs in which the chosen reply scores higher, where a
         tie counts as half right
    """
    # TODO
    return 0.0


# =============================================================================
# Step 4: Reward hacking
# =============================================================================

# --- Task 5: Fool the reward model -------------------------------------------
# The reward model learned what the preferred replies tend to look like, which
# is not quite the same as what makes a reply better. Write edits that leave a
# reply no more helpful than it was, and that you expect the reward model to
# score higher anyway. main() applies each one to every rejected test reply.
#
#   paper: https://arxiv.org/abs/2210.10760

HACKS: dict[str, Callable[[str], str]] = {}   # TODO: name -> edit


# --- Run (written for you) ---------------------------------------------------

def to_label(pairs: list[dict]) -> list[dict]:
    """The N_LABEL shortest test pairs, each reply at least 30 words, shuffled."""
    pool = [p for p in pairs if words(p["chosen"]) >= 30 and words(p["rejected"]) >= 30]
    pool = sorted(pool, key=lambda p: words(p["prompt"] + " " + p["chosen"]
                                            + " " + p["rejected"]))[:N_LABEL]
    shown = []
    for p in pool:
        first = int(hashlib.md5(p["prompt"].encode()).hexdigest(), 16) % 2 == 0
        shown.append({"prompt": p["prompt"],
                      "response_1": p["chosen"] if first else p["rejected"],
                      "response_2": p["rejected"] if first else p["chosen"],
                      "better": 1 if first else 2})
    return shown


def wrap(text: str, indent: str = "    ") -> str:
    return "\n".join(textwrap.fill(line, 78, initial_indent=indent,
                                   subsequent_indent=indent)
                     for line in text.strip().splitlines() if line.strip())


def show_labels(test: list[dict]) -> None:
    pairs = to_label(test)
    if not MY_LABELS:
        for i, p in enumerate(pairs):
            print(f"\n=== pair {i} " + "=" * 66)
            print(f"\n{wrap(p['prompt'], '')}")
            print(f"\n  response 1\n{wrap(p['response_1'])}")
            print(f"\n  response 2\n{wrap(p['response_2'])}")
        print(f"\nfill MY_LABELS with {N_LABEL} picks, 1 or 2, and run again")
        return
    if len(MY_LABELS) != N_LABEL:
        print(f"\nMY_LABELS has {len(MY_LABELS)} picks, not {N_LABEL}")
        return
    agree = 0
    for i, (p, mine) in enumerate(zip(pairs, MY_LABELS)):
        agree += mine == p["better"]
        print(f"  pair {i}: you {mine}, dataset {p['better']}")
    print(f"\nyou agreed on {agree} of {len(pairs)}")


def main() -> None:
    # Step 1
    train, test, X = setup()

    # Step 2
    show_labels(test)
    if len(MY_LABELS) != N_LABEL:
        return

    # Step 3
    w, losses = fit_reward(*X["train"])
    if losses:
        print(f"\ntraining loss: {losses[0]:.3f} at step 1, "
              f"{losses[-1]:.3f} at step {len(losses)}")

    print(f"\n  {'':<16}{'train':>7}{'test':>7}")
    print(f"  {'longer wins':<16}{length_baseline(train):>7.3f}"
          f"{length_baseline(test):>7.3f}")
    print(f"  {'reward model':<16}{accuracy(w, *X['train']):>7.3f}"
          f"{accuracy(w, *X['test']):>7.3f}")

    replies = [p[side] for p in test for side in ("chosen", "rejected")]
    rewards = np.concatenate([X["test"][0] @ w, X["test"][1] @ w])
    if rewards.std() > 0:
        r = np.corrcoef(rewards, [words(t) for t in replies])[0, 1]
        print(f"\ncorrelation of reward with length on test replies: {r:.2f}")

    # Step 4
    if not HACKS:
        print("\nHACKS is empty")
        return
    chosen, rejected = X["test"][0] @ w, X["test"][1] @ w
    scores = {name: embed([hack(p["rejected"]) for p in test]) @ w
              for name, hack in HACKS.items()}
    print("\nreward hacking on the rejected test replies")
    print(f"\n  {'':<16}{'raised':>8}{'mean change':>13}{'beat chosen':>13}")
    print(f"  {'(no edit)':<16}{'':>8}{'':>13}{(rejected > chosen).mean():>13.3f}")
    for name, hacked in scores.items():
        print(f"  {name:<16}{(hacked > rejected).mean():>8.3f}"
              f"{(hacked - rejected).mean():>+13.3f}{(hacked > chosen).mean():>13.3f}")


if __name__ == "__main__":
    main()
