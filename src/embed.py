"""
embed.py
--------
Embedding helpers shared by the baseline and the custom models.

    model = load_st_model()                       # off-the-shelf all-MiniLM-L6-v2
    emb   = embed_texts(model, texts)             # (n, 384), L2-normalized
    pca   = fit_pca(emb[train_mask], dim=64)      # fit on TRAIN chunks only
    small = apply_pca(pca, emb)                   # (n, 64), L2-normalized

Dependencies: pip install sentence-transformers scikit-learn numpy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

BASELINE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_st_model(name: str = BASELINE_MODEL, max_seq_length: int = 128, device: str | None = None):
    """Load a SentenceTransformer (local path or HF name). Imports are lazy so other modules stay light."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(name, device=device)
    model.max_seq_length = max_seq_length
    return model


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def embed_texts(model, texts: list[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    """Encode texts -> float32 numpy array. Normalized so dot product == cosine similarity."""
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    )
    return emb.astype(np.float32)


# --------------------------------------------------------------------------- #
# Dimensionality reduction (gives the baseline the same dimension as the custom model)
# --------------------------------------------------------------------------- #
def fit_pca(train_emb: np.ndarray, dim: int = 64, seed: int = 42) -> PCA:
    dim = int(min(dim, train_emb.shape[0] - 1, train_emb.shape[1]))
    return PCA(n_components=dim, random_state=seed).fit(train_emb)


def apply_pca(pca: PCA, emb: np.ndarray) -> np.ndarray:
    return l2_normalize(pca.transform(emb)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #
def save_embeddings(path: Path, emb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)
