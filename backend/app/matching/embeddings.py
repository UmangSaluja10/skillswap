"""Text -> vector, with a disk cache.

Two embedders behind one interface:
  * MiniLM  - sentence-transformers `all-MiniLM-L6-v2` (about 90 MB, runs offline after the first
              download). The real thing; used automatically when the package is installed.
  * Hashing - tiny built-in fallback (no download, no install). Understands shared words and the
              project's alias list (swiftui -> iOS), so the app still works and tests run offline.

`EMBEDDER=auto|minilm|hash` in .env picks one (auto = MiniLM if available, else hashing).
Each embedder states a calibration range so "semantic similarity" means the same 0..1 either way.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from ..resume.taxonomy import canonical_skill, normalize_domain

log = logging.getLogger("skillswap.matching")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    name: str
    dim: int
    calib: tuple[float, float]  # cosine range that maps to similarity 0..1

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _unit(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class HashingEmbedder:
    name = "hash-384"
    dim = 384
    calib = (0.02, 0.40)

    def _features(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9+#]+", text.lower())
        feats = list(words)
        feats += [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for w in words:
            domain = normalize_domain(w)
            if domain:
                feats += [f"dom::{domain}"] * 2   # "swiftui", "figma" ... share a domain feature
            canon = canonical_skill(w)
            if canon.lower() != w:
                feats.append(f"skill::{canon.lower()}")
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for f in self._features(text):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _unit(out)


class MiniLMEmbedder:
    name = "minilm-l6-v2"
    dim = 384
    calib = (0.10, 0.60)

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, only when used
        self._model = SentenceTransformer(MODEL_NAME)

    def embed(self, texts: list[str]) -> np.ndarray:
        return _unit(np.asarray(self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False),
                                dtype=np.float32))


def get_embedder(choice: str | None = None) -> Embedder:
    choice = (choice or os.environ.get("EMBEDDER", "auto")).strip().lower()
    if choice == "hash":
        return HashingEmbedder()
    try:
        return MiniLMEmbedder()
    except Exception as exc:  # not installed, or model not downloaded and no internet
        if choice == "minilm":
            raise
        log.warning("MiniLM unavailable (%s); using the built-in hashing embedder.", str(exc)[:120])
        return HashingEmbedder()


class EmbeddingStore:
    """Embeds texts once; the same text never costs a second computation (kept on disk)."""

    def __init__(self, embedder: Embedder | None = None, path: str | Path | None = ":memory:") -> None:
        self.embedder = embedder or get_embedder()
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False) if path else None
        if self._db is not None:
            self._db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.embedder.name}\n{text}".encode()).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        found: dict[str, np.ndarray] = {}
        with self._lock:
            if self._db is not None:
                for k in set(keys):
                    row = self._db.execute("SELECT v FROM emb WHERE k=?", (k,)).fetchone()
                    if row:
                        found[k] = np.frombuffer(row[0], dtype=np.float32)
            todo = {k: t for k, t in zip(keys, texts) if k not in found}
            if todo:
                fresh = self.embedder.embed(list(todo.values()))
                for (k, _), vec in zip(todo.items(), fresh):
                    found[k] = vec.astype(np.float32)
                    if self._db is not None:
                        self._db.execute("INSERT OR REPLACE INTO emb VALUES (?, ?)", (k, found[k].tobytes()))
                if self._db is not None:
                    self._db.commit()
        return np.stack([found[k] for k in keys]) if keys else np.zeros((0, self.embedder.dim), np.float32)

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.embed([a, b])
        return float(np.dot(va, vb))

    def calibrated(self, cosine: float) -> float:
        lo, hi = self.embedder.calib
        return float(min(1.0, max(0.0, (cosine - lo) / (hi - lo))))
