"""
evaluate.py
-----------
Plot helpers (more evaluation functions - similarity stats, neighbor checks - get added here later).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_k_sweep(sweep: dict, chosen_k: int, title: str, path: Path) -> None:
    """Silhouette (higher = better) and Davies-Bouldin (lower = better) vs k."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(sweep["k"], sweep["silhouette"], "o-", color="tab:blue", label="silhouette (cosine)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("silhouette", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(sweep["k"], sweep["davies_bouldin"], "s--", color="tab:orange", label="Davies-Bouldin")
    ax2.set_ylabel("Davies-Bouldin", color="tab:orange")
    ax1.axvline(chosen_k, color="gray", linestyle=":", label=f"chosen k={chosen_k}")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def project_2d(emb: np.ndarray, seed: int = 42) -> np.ndarray:
    perplexity = max(2, min(30, (emb.shape[0] - 1) // 3))
    return TSNE(n_components=2, init="pca", perplexity=perplexity, random_state=seed).fit_transform(emb)


def plot_cluster_map(xy: np.ndarray, labels: np.ndarray, doc_ids: list[str], title: str, path: Path) -> None:
    """Two panels with the same 2D layout: colored by predicted cluster, and colored by source document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab20", s=12)
    axes[0].set_title(f"{title} - by cluster")

    uniq = sorted(set(doc_ids))
    ids = np.array(doc_ids)
    cmap = plt.get_cmap("tab20")
    for i, d in enumerate(uniq):
        m = ids == d
        axes[1].scatter(xy[m, 0], xy[m, 1], color=cmap(i % 20), s=12, label=d)
    axes[1].set_title(f"{title} - by source document")
    if len(uniq) <= 12:
        axes[1].legend(fontsize=7, loc="best")
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss_curve(history: dict, title: str, path: Path) -> None:
    """Train vs held-out (val) InfoNCE loss per epoch. Epoch 0 = before any training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ep = history["epoch"]
    tr = [(e, v) for e, v in zip(ep, history["train_loss"]) if v is not None]
    va = [(e, v) for e, v in zip(ep, history["val_loss"]) if v is not None]
    if tr:
        ax.plot(*zip(*tr), "o-", label="train (train docs)")
    if va:
        ax.plot(*zip(*va), "s--", label="val (held-out docs)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("InfoNCE loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
