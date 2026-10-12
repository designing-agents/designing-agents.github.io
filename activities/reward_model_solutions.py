# Reward Models - SOLUTION KEY.

from __future__ import annotations

import hashlib
import json
import re
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


# --- Setup -------------------------------------------------------------------

def download() -> list[dict]:
    path = hf_hub_download(DATASET, DATA_FILE, repo_type="dataset")
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_pairs(rows: list[dict]) -> list[dict]:
    return [{"prompt": r["question"], "chosen": r["chosen"],
             "rejected": r["rejected"]} for r in rows]


def embed(texts: list[str]) -> np.ndarray:
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
    rows = download()
    train = load_pairs(rows[:N_TRAIN])
    test = load_pairs(rows[N_TRAIN:N_TRAIN + N_TEST])
    X = {name: (embed([p["chosen"] for p in pairs]),
                embed([p["rejected"] for p in pairs]))
         for name, pairs in (("train", train), ("test", test))}
    print(f"setup done: {len(train)} train pairs, {len(test)} test pairs")
    return train, test, X


# --- Task 1 ------------------------------------------------------------------

MY_LABELS: list[int] = [1, 2, 1, 2, 1, 2, 1, 2]   # any picks are right


# --- Task 2 ------------------------------------------------------------------

def length_baseline(pairs: list[dict]) -> float:
    if not pairs:
        return 0.0
    score = 0.0
    for p in pairs:
        c, r = words(p["chosen"]), words(p["rejected"])
        score += 1.0 if c > r else 0.5 if c == r else 0.0
    return score / len(pairs)


# --- Task 3 ------------------------------------------------------------------

def fit_reward(X_chosen: np.ndarray, X_rejected: np.ndarray, lr: float = LR,
               epochs: int = EPOCHS) -> tuple[np.ndarray, list[float]]:
    Xc = torch.tensor(X_chosen, dtype=torch.float32)
    Xr = torch.tensor(X_rejected, dtype=torch.float32)
    w = torch.zeros(Xc.shape[1], requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=lr)
    losses = []
    for _ in range(epochs):
        # -log sigmoid(r_chosen - r_rejected), averaged over pairs. logsigmoid
        # avoids log(0) when a margin is large and negative. At w = 0 the first
        # loss is log 2, about 0.693.
        loss = -torch.nn.functional.logsigmoid((Xc - Xr) @ w).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return w.detach().numpy(), losses


# --- Task 4 ------------------------------------------------------------------

def accuracy(w: np.ndarray, X_chosen: np.ndarray,
             X_rejected: np.ndarray) -> float:
    margin = X_chosen @ w - X_rejected @ w
    return float(((margin > 0) + 0.5 * (margin == 0)).mean())


# --- Task 5 ------------------------------------------------------------------

def _first_half(reply: str) -> str:
    w = reply.split()
    return " ".join(w[: max(1, len(w) // 2)])


def _drop_opener(reply: str) -> str:
    # Removes the first sentence, e.g. "Sure, I'd be happy to help!"
    parts = re.split(r"(?<=[.!?])\s+", reply.strip(), maxsplit=1)
    return parts[1] if len(parts) == 2 else reply


def _both(reply: str) -> str:
    return _first_half(_drop_opener(reply))


def _pleasantries(reply: str) -> str:
    # Should lower the reward: Llama-2's replies open and close like this.
    return ("Great question! " + reply.strip()
            + "\n\nI hope this helps! Let me know if you have any other questions.")


HACKS: dict[str, Callable[[str], str]] = {
    "first half": _first_half,
    "drop opener": _drop_opener,
    "both": _both,
    "pleasantries": _pleasantries,
}


# --- Run ---------------------------------------------------------------------

def to_label(pairs: list[dict]) -> list[dict]:
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
    train, test, X = setup()

    show_labels(test)
    if len(MY_LABELS) != N_LABEL:
        return

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
