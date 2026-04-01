from __future__ import annotations


class SimpleTokenizer:
    """Minimal tokenizer stub for project bring-up."""

    def __init__(self, vocab_size: int = 32000) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [ord(ch) % self.vocab_size for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id % 256) for token_id in token_ids)
