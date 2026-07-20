from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np


class TextEncoder(Protocol):
    dimension: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashTextEncoder:
    """Deterministic character n-gram encoder for offline tests and smoke runs."""

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension
        self.identifier = f"hash-char-ngram-{dimension}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        output = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            normalized = re.sub(r"\s+", " ", text.strip().lower())
            grams = [normalized[i : i + n] for n in (1, 2, 3) for i in range(max(0, len(normalized) - n + 1))]
            for gram in grams or ["<empty>"]:
                raw = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(raw, "little")
                output[row, value % self.dimension] += 1.0 if value & 1 else -1.0
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        output /= np.maximum(norms, 1e-12)
        return output


class SentenceTransformerTextEncoder:
    def __init__(self, model_name: str, batch_size: int = 32, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the 'semantic' extra to use SentenceTransformer") from exc
        self.model = SentenceTransformer(model_name, device=device)
        self.identifier = model_name
        self.batch_size = batch_size
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = self.model.encode(
            list(texts), batch_size=self.batch_size, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True,
        )
        return np.asarray(values, dtype=np.float32)
