"""
Autoprompting - SOLUTION KEY.

    pip install google-genai
    export GEMINI_API_KEY=...
    python autoprompting_solutions.py

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

import warnings
warnings.filterwarnings("ignore")

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

# A topic with plenty of stock imagery gives the taxonomy something to catch.
TOPIC = "capitalism"
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


# --- Task 0 -------------------------------------------------------------------

# Deliberately plain. A seed that already says everything leaves the loop
# nothing to do, and the first poem gives Task 1 nothing to name.
GENERATOR_PROMPT = f"Write a short poem about {TOPIC}."


# --- Task 1 -------------------------------------------------------------------

# Each entry describes something a reader can point at in the text. "Not
# evocative enough" fails that test: two judges would disagree on it, and the
# rewrite step gets nothing it can act on.
ERROR_TAXONOMY: dict[str, str] = {
    "cliche": "a stock image or phrase that most poems on this topic use, such "
              "as silver light, a lonely night, or the moon as a watchful eye",
    "forced rhyme": "a word chosen to rhyme that is awkward, archaic, or "
                    "makes the line say less than it would otherwise",
    "sing-song meter": "every line falls into the same bouncing rhythm, so "
                       "the poem reads like a greeting card",
    "stated feeling": "the poem names an emotion, such as longing, peace, or "
                      "wonder, without anything in it that causes the emotion",
    "moral ending": "the last lines sum up a lesson or a general truth "
                    "instead of ending on something concrete",
    "padding": "a line that restates an earlier line without adding anything",
}


# --- Task 2 -------------------------------------------------------------------

JUDGE_PROMPT = """Here is a poem about "{topic}":

{poem}

A poem can have these errors:
{taxonomy}

List the name of every error this poem has, one per line, and nothing else.
If it has none, reply NONE."""


def judge(poem: str) -> list[str]:
    taxonomy = "\n".join(f"- {name}: {description}"
                         for name, description in ERROR_TAXONOMY.items())
    reply = ask(JUDGE_PROMPT.format(topic=TOPIC, poem=poem, taxonomy=taxonomy))
    # The judge echoes the list format it was shown, so lines tend to arrive as
    # "- cliche" or "**Cliche**: ...". Comparing raw lines against the names
    # drops every real hit. Strip the decoration, cut at a colon, and match
    # without case.
    names = {name.lower(): name for name in ERROR_TAXONOMY}
    found = []
    for line in reply.splitlines():
        line = line.split(":")[0].strip().strip("-*•`\"' ").lower()
        if line in names and names[line] not in found:
            found.append(names[line])
    return found


# --- Task 3 -------------------------------------------------------------------

REWRITE_PROMPT = """This prompt was given to a language model to write a poem \
about "{topic}":

{prompt}

It wrote this poem:

{poem}

A reviewer found these problems in the poem:
{errors}

Write an improved prompt that would avoid these problems while keeping what \
works. Reply with the new prompt only: no explanation, no title, no quotation \
marks."""


def rewrite(prompt: str, poem: str, errors: list[str]) -> str:
    # Nothing to fix. Returning the prompt unchanged means every later round
    # repeats this one from the cache, which is the right answer: the loop has
    # converged as far as this judge can tell.
    if not errors:
        return prompt
    # The descriptions go in, not just the names. "padding" alone tells the
    # rewriter much less than what padding looks like.
    listed = "\n".join(f"- {name}: {ERROR_TAXONOMY[name]}" for name in errors)
    reply = ask(REWRITE_PROMPT.format(topic=TOPIC, prompt=prompt, poem=poem,
                                      errors=listed))
    # Asked for the prompt alone, the model still sometimes wraps it in a code
    # fence or quotes. Whatever is left becomes the next round's prompt word
    # for word, so a stray "Here is the improved prompt:" would be sent to the
    # poet too.
    new = reply.strip()
    if new.startswith("```"):
        new = new.split("\n", 1)[-1].rsplit("```", 1)[0]
    new = new.strip().strip('"').strip()
    return new or prompt


# --- Task 4 -------------------------------------------------------------------

def autoprompt(seed: str, rounds: int = ROUNDS) -> list[dict]:
    history, prompt = [], seed
    for i in range(rounds + 1):
        poem = ask(prompt, temperature=1.0)
        errors = judge(poem)
        history.append({"prompt": prompt, "poem": poem, "errors": errors})
        # rounds + 1 poems need only rounds rewrites. A rewrite after the last
        # poem costs a request and its prompt is never used.
        if i < rounds:
            prompt = rewrite(prompt, poem, errors)
    return history


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
