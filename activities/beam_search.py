"""
Beam Search

Greedy decoding takes the best token at each step, which is not the same as
finding the best overall sequence.

    pip install transformers torch
    python beam_search.py
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MODEL = "gpt2"
MAX_NEW_TOKENS = 30          # a cap; most beams end their sentence before it
BEAM_WIDTHS = (1, 2, 4)   # a width of one is greedy, and the baseline here
ALPHA = 0.7                  # how hard Task 4 corrects for length

PROMPTS = {
    "open-ended": "The support agent opened the ticket, read the error log, and",
    "factual":    "The largest planet in the solar system is",
    "creative":   "Once upon a time there was a",
}


@dataclass
class Beam:
    """One candidate sequence and the score it has accumulated."""

    ids: list[int]          # the prompt's tokens, then everything generated
    logprob: float = 0.0    # summed over the generated tokens only
    done: bool = False      # True once this beam has finished its sentence


def is_finished(tokenizer, token_id: int) -> bool:
    """Written for you. Used by Task 2."""
    if token_id == tokenizer.eos_token_id:
        return True
    return tokenizer.decode([token_id]).rstrip().endswith((".", "!", "?"))


# --- Task 0: Load ------------------------------------------------------------
# gpt2 is small enough to run on a laptop CPU.
# Running text through it returns a score for every token in the vocabulary at
# every position of the input.
#
#   auto classes: https://huggingface.co/docs/transformers/model_doc/auto
#   generation:   https://huggingface.co/docs/transformers/llm_tutorial

def load_model() -> tuple:
    """
    In:  nothing
    Out: (model, tokenizer) for MODEL, with the model in evaluation mode
    """
    # TODO
    return None, None


# --- Task 1: Score a sequence ------------------------------------------------
# A search needs something to maximize. The score of a sequence is the log
# probability the model gives it: the sum, over positions, of the log
# probability of the token that actually came next.
#
#   log softmax: https://pytorch.org/docs/stable/nn.functional.html

def sequence_logprob(model, tokenizer, ids: list[int]) -> float:
    """
    In:  a model, its tokenizer, and a list of token ids
    Out: the total log probability of every token after the first

    The first token has nothing before it to condition on, so it scores
    nothing.
    """
    # TODO
    return 0.0


# --- Task 2: One step of the search ------------------------------------------
# One step takes every unfinished beam, asks the model what could come next, and
# returns every pairing worth considering.

def expand(model, tokenizer, beams: list[Beam], k: int) -> list[Beam]:
    """
    In:  a model, its tokenizer, the current beams, and how many continuations
         to take from each one
    Out: the candidate beams after one step, before any pruning

    A finished beam comes back unchanged rather than being extended again.
    """
    # TODO
    return beams


# --- Task 3: The search ------------------------------------------------------
# Start with one beam holding nothing but the prompt. Then repeat: expand every
# beam, and keep only the best beam_width candidates. This pruning is what stops
# the search multiplying by k at every step.

def beam_search(model, tokenizer, prompt: str, beam_width: int,
                max_new_tokens: int = MAX_NEW_TOKENS) -> list[Beam]:
    """
    In:  a model, its tokenizer, a prompt, how many beams to keep, and the most
         tokens to add
    Out: the final beams, highest scoring first
    """
    # TODO
    return []


# --- Task 4: Length ----------------------------------------------------------
# Run Task 3 first and read the beams it prints. Every token added lowers a
# sequence's score, so a beam that ends its sentence early beats one that keeps
# going, whatever the two actually say. Dividing the score by the length corrects
# for this. Raising the length to a power between 0 and 1 first lets you choose
# how hard to correct.
#
#   Paper: https://arxiv.org/abs/1609.08144

def length_normalize(logprob: float, length: int, alpha: float = ALPHA) -> float:
    """
    In:  a beam's log probability, how many tokens it generated, and an exponent
    Out: the score adjusted for length

    An alpha of 0 leaves the score alone, and an alpha of 1 divides by the full
    length. A length of 0 leaves the score alone.
    """
    # TODO
    return logprob


# --- Task 5: Pick a winner ---------------------------------------------------
# Task 3 returns the beams sorted by raw score. Normalizing can reorder them,
# so return the highest-scoring beam AFTER length normalization.

def best_beam(beams: list[Beam], n_prompt: int, alpha: float = 0.0) -> Beam:
    """
    In:  the beams a search returned, how many of their tokens are prompt, and
         a normalization exponent
    Out: the beam scoring highest under that normalization
    """
    # TODO
    return beams[0]


# --- Run (written for you) ---------------------------------------------------
# Every beam is listed with its raw score, so the top line is what the search
# picks on its own. WINNER is what Task 5 picks after normalizing for length.
# When the two differ, that is Task 4 doing its work.

def main() -> None:
    model, tokenizer = load_model()
    if model is None:
        print("load_model returned nothing")
        return

    for name, prompt in PROMPTS.items():
        prompt_ids = tokenizer.encode(prompt)
        n_prompt = len(prompt_ids)
        text = lambda beam: tokenizer.decode(
            beam.ids[n_prompt:]).strip().replace("\n", " ")

        print(f"\n{name}: {prompt!r}")
        top = None
        for width in BEAM_WIDTHS:
            beams = beam_search(model, tokenizer, prompt, width)
            print(f"\n  beam {width}")
            if not beams:
                print("    beam_search returned nothing")
                continue
            for beam in beams:
                print(f"    {beam.logprob:>8.2f}  {text(beam)}")
            print(f"    WINNER: {text(best_beam(beams, n_prompt, ALPHA))}")
            top = beams[0]

        # Task 1 checking Tasks 2 and 3. A beam's running score should equal the
        # score of the whole sequence less the score of the prompt on its own.
        if top is not None:
            recomputed = (sequence_logprob(model, tokenizer, top.ids)
                          - sequence_logprob(model, tokenizer, prompt_ids))
            print(f"\n  running score {top.logprob:.3f}, "
                  f"recomputed {recomputed:.3f}")


if __name__ == "__main__":
    main()
