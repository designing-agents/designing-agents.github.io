"""
Beam Search - SOLUTION KEY. Do not distribute.

    pip install transformers torch
    python beam_search_solutions.py

Takes a minute or two on a laptop. There is no KV cache, so every step re-reads
the whole prefix, and the four widths together run the model fifteen times per
step per prompt.

None of this has been run against gpt2: its weights would not download on the
machine it was written on. The scores and the sentences are unverified, and so
is the claim that the winning sentence dulls as the beam widens. Run it once
before class.
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
    # eval() only matters for dropout and batchnorm, which gpt2 inference does
    # not hit, but no_grad() in _next_logprobs does matter: without it torch
    # builds a graph for every step of every beam and the run crawls.
    model.eval()
    return model, tokenizer


# --- Task 1 -------------------------------------------------------------------

def sequence_logprob(model, tokenizer, ids: list[int]) -> float:
    if len(ids) < 2:
        return 0.0
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0]
    # Row i predicts token i + 1, so the last row predicts nothing that is in
    # the sequence and is dropped. Off-by-one here is the usual mistake, and it
    # shows up as the check at the bottom of the file disagreeing by exactly one
    # token's worth of score.
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
            # beam.ids + [token_id] builds a new list. Appending to beam.ids in
            # place is the bug that costs the most time here: every sibling
            # shares the one list, so they all grow together and the search
            # quietly returns k copies of the same sequence.
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
        # The student file no longer says what k to pass. beam_width is the
        # answer, and beam_width continuations from each of beam_width beams is
        # the most that can matter: a candidate ranked below k within its own
        # parent has k better siblings and cannot reach the top k overall.
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


# --- Notes for discussion -----------------------------------------------------
# Width 1 is greedy, and a wider beam can never score below it, because the
# wider search considers everything the narrower one did. So the interesting
# question is never whether beam search wins on probability, only whether the
# sentence it wins with is better. It is not: on the open-ended and creative
# prompts the winner should get blander and more repetitive as the width grows.
# The most likely sequence is not the best sequence, which is the point of the
# activity and the reason HW2 samples instead.
# (Holtzman et al., https://arxiv.org/abs/1904.09751)
#
# Where WINNER is not the top line: before normalizing, the search prefers
# whichever beam ends its sentence soonest, because every extra token subtracts
# log probability. The top line tends to be short and abrupt and WINNER longer.
# That is not a bug in their code, it is Task 4's reason to exist.
#
# Expect the prompts to split. The factual one has roughly one right answer, so
# the likeliest continuation is the wanted one. The open-ended and creative
# prompts have many acceptable continuations, and picking the likeliest is what
# makes them dull. That is also why translation and summarization use beam
# search while chat assistants sample.
#
# Three things worth having ready:
#
# Width 1 matches greedy only when no two logits tie exactly. argmax takes the
# first of equal values and topk gives no such guarantee, so on a toy model with
# hand-set probabilities the two can part ways. gpt2 logits come off a real
# forward pass and do not tie.
#
# The length bias only appears because is_finished stops at sentence end. If
# beams could stop only at <|endoftext|> they would all run to MAX_NEW_TOKENS,
# and dividing equal lengths by length ** alpha would reorder nothing.
#
# The claim about the sentence dulling is expected, not observed - this has
# never been run against gpt2. Check it before class.


if __name__ == "__main__":
    main()
