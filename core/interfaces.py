from typing import Protocol

from core.schemas import SearchResult


class Retriever(Protocol):
    """Anything that can rank chunks for a query."""

    def search(self, query: str, k: int = 5) -> list[SearchResult]: ...


class LLM(Protocol):
    """Anything that can turn instructions + a user message into text."""

    def generate(self, system: str, user: str, temperature: float = 0.0) -> str: ...


class VLM(Protocol):
    """Anything that can answer a question about an image."""

    def ask_image(self, image_png: bytes, question: str) -> str: ...
