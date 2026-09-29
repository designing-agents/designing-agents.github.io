"""
Beam Search - SOLUTIONS.

    pip install transformers torch
    python beam_search_solutions.py
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"
MAX_NEW_TOKENS = 30          # a cap; most beams end their sentence before it
BEAM_WIDTHS = (1, 2, 4)      # a width of one is greedy, and the baseline here
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


def _next_logprobs(model, ids: list[int]) -> torch.Tensor:
    """
    Log probabilities for the token after `ids`. Not in the student file; they
    can inline it into Task 2.
    """
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
    return torch.log_softmax(logits, dim=-1)


# --- Task 0 -------------------------------------------------------------------

def load_model() -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()
    return model, tokenizer


# --- Task 1 -------------------------------------------------------------------

def sequence_logprob(model, tokenizer, ids: list[int]) -> float:
    if len(ids) < 2:
        return 0.0
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0]
    logprobs = torch.log_softmax(logits[:-1], dim=-1)
    targets = torch.tensor(ids[1:]).unsqueeze(1)
    return float(logprobs.gather(1, targets).sum())


# --- Task 2 -------------------------------------------------------------------

def expand(model, tokenizer, beams: list[Beam], k: int) -> list[Beam]:
    candidates = []
    for beam in beams:
        if beam.done:
            candidates.append(beam)
            continue
        top = torch.topk(_next_logprobs(model, beam.ids), k)
        for logprob, token_id in zip(top.values, top.indices):
            token_id = int(token_id)
            candidates.append(Beam(beam.ids + [token_id],
                                   beam.logprob + float(logprob),
                                   is_finished(tokenizer, token_id)))
    return candidates


# --- Task 3 -------------------------------------------------------------------

def beam_search(model, tokenizer, prompt: str, beam_width: int,
                max_new_tokens: int = MAX_NEW_TOKENS) -> list[Beam]:
    beams = [Beam(tokenizer.encode(prompt))]
    for _ in range(max_new_tokens):
        if all(beam.done for beam in beams):
            break
        candidates = expand(model, tokenizer, beams, beam_width)
        beams = sorted(candidates, key=lambda b: b.logprob,
                       reverse=True)[:beam_width]
    return sorted(beams, key=lambda b: b.logprob, reverse=True)


# --- Task 4 -------------------------------------------------------------------

def length_normalize(logprob: float, length: int, alpha: float = ALPHA) -> float:
    if length == 0:
        return logprob
    return logprob / (length ** alpha)


# --- Task 5 -------------------------------------------------------------------

def best_beam(beams: list[Beam], n_prompt: int, alpha: float = 0.0) -> Beam:
    return max(beams, key=lambda b: length_normalize(b.logprob,
                                                     len(b.ids) - n_prompt,
                                                     alpha))


# --- Run (written for you) ---------------------------------------------------
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

        if top is not None:
            recomputed = (sequence_logprob(model, tokenizer, top.ids)
                          - sequence_logprob(model, tokenizer, prompt_ids))
            print(f"\n  running score {top.logprob:.3f}, "
                  f"recomputed {recomputed:.3f}")


if __name__ == "__main__":
    main()
