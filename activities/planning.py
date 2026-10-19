"""In-class activity: prompt routing, chaining, and planning."""

import os
from google import genai
from typing import List, Optional


def query_gemini(prompt: str, temperature: float = 0.0) -> str:
    # implement
    raise NotImplementedError


def has_won(board: List[str], symbol: str) -> bool:
    lines = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ]
    return any(board[a] == board[b] == board[c] == symbol for a, b, c in lines)


def show_game(board: List[str]) -> None:
    """TODO: print the board."""
    raise NotImplementedError


def human_turn(board: List[str], human_symbol: str = "X") -> None:
    """TODO: prompt user, validate move, place symbol."""
    raise NotImplementedError


def agent_turn(board: List[str], human_symbol: str = "X", agent_symbol: str = "O") -> Optional[str]:
    """
    TODO: implement this exact 4-step agent flow.

    Step 1 (deterministic check):
    - Observe the board and check if the human has won using `has_won`.
    - If yes, return "Congrats! Good game".

    Step 2 (LLM call #1):
    - If the human has not won, ask Gemini for possible places to play.
    - Parse/validate those moves against the board state.

    Step 3 (LLM call #2):
    - Ask Gemini to choose the best move from the possible places.
    - Apply that move for `agent_symbol`.

    Step 4 (deterministic check):
    - After playing, check if the agent has won using `has_won`.
    - If yes, return "I beat you!".

    Return None if the game should continue, or "Draw." if no moves remain.
    """
    raise NotImplementedError


def play_game() -> None:
    """TODO: run the game loop."""
    raise NotImplementedError


if __name__ == "__main__":
    play_game()
