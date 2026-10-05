"""
Autoprompting

Rather than rewriting a prompt by hand until the output looks right, have the
model rewrite it. You will pick a one-word topic, write a prompt asking for a
poem about it, and list the ways those poems can be improved.

    pip install google-genai
    export GEMINI_API_KEY=...
    python autoprompting.py

Running the file prints every round's prompt, its poem, and the errors the
judge found.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from google import genai
from google.genai import types

# Ignore warnings from the Gemini SDK.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

MODEL = "gemini-3.5-flash-lite"
MAX_TOKENS = 1024
CACHE = Path(__file__).resolve().parent / ".autoprompting_cache.json"

TOPIC = ""      # TODO: one word for the poems to be about
ROUNDS = 3      # rewrites to try

_client = None


def ask(prompt: str, temperature: float = 0.0) -> str:
    """
    Written for you. Sends one prompt to MODEL and returns the reply text.

    Every reply is saved to .autoprompting_cache.json, and an identical request
    returns the saved reply, even at a temperature above 0. Delete the file to
    draw fresh replies. Waits and retries when the free tier's rate limit is
    hit.
    """
    global _client
    request = {"model": MODEL, "prompt": prompt, "temperature": temperature}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    if key in cache:
        return cache[key]
    if _client is None:
        _client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=8)))
    response = _client.models.generate_content(
        model=MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature,
                                           max_output_tokens=MAX_TOKENS))
    reply = response.text or ""
    cache[key] = reply
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return reply


# --- Task 0: A first prompt --------------------------------------------------
# The prompt you would write by hand, asking for a poem about TOPIC.
#
#   prompting: https://ai.google.dev/gemini-api/docs/prompting-strategies

GENERATOR_PROMPT = ""   # TODO


# --- Task 1: Error taxonomy --------------------------------------------------
# Run the file with Task 0 done and read the poem it prints. Name each way it
# falls short, and describe it precisely enough that someone who has not seen
# this poem could spot it in a new one.
#
#   paper: https://arxiv.org/abs/2404.12272

ERROR_TAXONOMY: dict[str, str] = {}   # TODO: error name -> what it looks like


# --- Task 2: Judge -----------------------------------------------------------
# The judge prompt is written for you. Fill in its three blanks, send it at
# temperature 0, and read the error names out of the reply.
#
#   paper: https://arxiv.org/abs/2306.05685

JUDGE_PROMPT = """Here is a poem about "{topic}":

{poem}

A poem can have these errors:
{taxonomy}

List the name of every error this poem has, one per line, and nothing else.
If it has none, reply NONE."""


def judge(poem: str) -> list[str]:
    """
    In:  a poem about TOPIC
    Out: the names of the ERROR_TAXONOMY errors the judge found in it, and []
         when there are none

    Drop any name the judge returns that is not in ERROR_TAXONOMY.
    """
    # TODO
    return []


# --- Task 3: Rewrite ---------------------------------------------------------
# Show the model the current prompt, the poem it wrote, and the errors the
# judge found, and ask for a better prompt.

def rewrite(prompt: str, poem: str, errors: list[str]) -> str:
    """
    In:  the current prompt, its poem, and the errors found in the poem
    Out: a new prompt, with nothing around it
    """
    # TODO
    return prompt


# --- Task 4: The loop --------------------------------------------------------
# Each round, write a poem with the current prompt at temperature 1.0, judge
# it, and rewrite the prompt for the next round.

def autoprompt(seed: str, rounds: int = ROUNDS) -> list[dict]:
    """
    In:  a seed prompt and how many rewrites to try
    Out: one record per round with "prompt", "poem", and "errors", starting
         with the seed, so rounds + 1 records in all
    """
    # TODO
    return []


# --- Run (written for you) ---------------------------------------------------

def show(i: int, record: dict) -> None:
    print(f"\n=== round {i} " + "=" * 64)
    print(f"\n{record['prompt']}\n\n{record['poem']}")
    print(f"\nerrors: {', '.join(record['errors']) or 'none'}")


def main() -> None:
    if not TOPIC or not GENERATOR_PROMPT:
        print("TOPIC or GENERATOR_PROMPT is empty")
        return

    if not ERROR_TAXONOMY:
        print(f"\n{ask(GENERATOR_PROMPT, temperature=1.0)}")
        print("\nwrite ERROR_TAXONOMY to judge this poem and run the loop")
        return

    history = autoprompt(GENERATOR_PROMPT)
    if not history:
        print("autoprompt returned nothing")
        return
    for i, record in enumerate(history):
        show(i, record)

    print(f"\n  {'round':>5}{'errors':>8}")
    for i, record in enumerate(history):
        print(f"  {i:>5}{len(record['errors']):>8}")
    best = min(range(len(history)), key=lambda i: len(history[i]["errors"]))
    print(f"\nfewest errors: round {best}")


if __name__ == "__main__":
    main()
