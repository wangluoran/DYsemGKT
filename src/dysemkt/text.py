from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


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


class APITextEncoder:
    """OpenAI-compatible embeddings API encoder with local per-text caching."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        dimension: int | None = None,
        request_dimensions: int | None = None,
        cache_dir: Path | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.model_name = model_name
        self.batch_size = batch_size
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self._dimension = dimension
        self.request_dimensions = request_dimensions
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.timeout = timeout
        self.max_retries = max_retries
        self.identifier = f"api:{self.base_url}/embeddings:{model_name}"
        if self._dimension is not None:
            self.identifier += f":dim-{self._dimension}"
        if self.request_dimensions is not None:
            self.identifier += f":request-dim-{self.request_dimensions}"
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, batch_size: int = 32) -> APITextEncoder:
        load_env_file()
        model_name = os.environ.get("DYSEMKT_API_MODEL") or os.environ.get("DYSEMKT_EMBEDDING_MODEL")
        if not model_name:
            raise RuntimeError("Set DYSEMKT_API_MODEL in .env to use API embeddings")
        dimension = _optional_int("DYSEMKT_API_OUTPUT_DIM")
        request_dimensions = _optional_int("DYSEMKT_API_REQUEST_DIMENSIONS")
        cache_dir_value = os.environ.get("DYSEMKT_API_CACHE_DIR")
        return cls(
            model_name=model_name,
            batch_size=batch_size,
            base_url=os.environ.get("DYSEMKT_API_BASE_URL", "https://api.openai.com/v1"),
            api_key_env=os.environ.get("DYSEMKT_API_KEY_ENV", "OPENAI_API_KEY"),
            dimension=dimension,
            request_dimensions=request_dimensions,
            cache_dir=Path(cache_dir_value) if cache_dir_value else None,
            timeout=_optional_float("DYSEMKT_API_TIMEOUT", 60.0),
            max_retries=_optional_int("DYSEMKT_API_MAX_RETRIES", 3) or 0,
        )

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("API embedding dimension is unknown until encode() has completed")
        return self._dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        output: list[list[float] | None] = [None] * len(texts)
        pending: list[tuple[int, str]] = []
        for index, text in enumerate(texts):
            cached = self._read_cache(text)
            if cached is None:
                pending.append((index, text))
            else:
                output[index] = cached
                self._validate_dimension(cached)

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            embeddings = self._embed_batch([text for _, text in batch])
            if len(embeddings) != len(batch):
                raise RuntimeError("embedding API returned a mismatched number of rows")
            for (index, text), embedding in zip(batch, embeddings):
                self._validate_dimension(embedding)
                self._write_cache(text, embedding)
                output[index] = embedding

        if any(value is None for value in output):
            raise RuntimeError("embedding API did not produce all requested rows")
        return np.asarray(output, dtype=np.float32)

    def _validate_dimension(self, embedding: Sequence[float]) -> None:
        size = len(embedding)
        if self._dimension is None:
            self._dimension = size
        elif size != self._dimension:
            raise RuntimeError(f"embedding dimension mismatch: expected {self._dimension}, got {size}")

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        payload: dict[str, object] = {"model": self.model_name, "input": list(texts)}
        if self.request_dimensions is not None:
            payload["dimensions"] = self.request_dimensions
        response = self._post_json("/embeddings", payload)
        data = response.get("data")
        if not isinstance(data, list):
            raise RuntimeError("embedding API response is missing data")
        ordered = sorted(data, key=lambda row: int(row.get("index", 0)) if isinstance(row, dict) else 0)
        embeddings = []
        for row in ordered:
            if not isinstance(row, dict) or not isinstance(row.get("embedding"), list):
                raise RuntimeError("embedding API response has an invalid embedding row")
            embeddings.append([float(value) for value in row["embedding"]])
        return embeddings

    def _post_json(self, path: str, payload: dict[str, object]) -> dict:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {self.api_key_env} to use API embeddings")
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                    return json.loads(handle.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= self.max_retries:
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"embedding API request failed with HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"embedding API request failed: {exc}") from exc
            time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("embedding API request failed after retries")

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256()
        digest.update(self.identifier.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        return digest.hexdigest()

    def _read_cache(self, text: str) -> list[float] | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{self._cache_key(text)}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        embedding = value.get("embedding") if isinstance(value, dict) else None
        return [float(item) for item in embedding] if isinstance(embedding, list) else None

    def _write_cache(self, text: str, embedding: Sequence[float]) -> None:
        if self.cache_dir is None:
            return
        path = self.cache_dir / f"{self._cache_key(text)}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"embedding": list(embedding)}, handle)


def _optional_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


def _optional_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc
