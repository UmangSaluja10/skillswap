"""Checks the embedding model on your machine (downloads MiniLM the first time, ~90 MB).

    python -m app.matching.check
"""
from __future__ import annotations

import logging

from .embeddings import EmbeddingStore, get_embedder


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    embedder = get_embedder()
    print(f"Embedder in use: {embedder.name}")
    if embedder.name.startswith("hash"):
        print("  (built-in fallback. For the real model run: pip install -r requirements-embeddings.txt)")
    store = EmbeddingStore(embedder, path=None)
    pairs = [("SwiftUI developer", "iOS app developer"),
             ("SwiftUI developer", "Flutter cross-platform mobile developer"),
             ("Figma UI designer", "user interface and prototyping"),
             ("Figma UI designer", "backend REST API developer")]
    for a, b in pairs:
        cos = store.similarity(a, b)
        print(f"  {store.calibrated(cos):4.2f} (cos {cos:5.2f})  {a}  <->  {b}")
    print("Expected: the first and third pairs score clearly higher than the second and fourth.")


if __name__ == "__main__":
    main()
