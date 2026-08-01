"""Abstract interface every state reader implementation must satisfy.

Keeping this interface tiny is the whole point: SekiroEnv and every other
downstream module depend only on `StateReader.read()` returning a
`GameState`, never on how the state was obtained. That lets stage 1-6 run
entirely against `MockStateReader` and stage 8 swap in `PixelStateReader`
without touching any other module.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .schema import GameState


class StateReader(ABC):
    @abstractmethod
    def read(self) -> GameState:
        """Return the current game state as a GameState instance."""
        raise NotImplementedError

    def is_available(self) -> bool:
        """Whether this reader can currently produce real data.

        Real readers (e.g. screen-based) should return False when the game
        window isn't found, so callers can decide to fall back to a mock.
        Mock readers are always available.
        """
        return True

    def close(self) -> None:
        """Release any resources (process handles, sockets, etc.)."""
        return None
