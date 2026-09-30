"""
run_baseline.py
---------------
Baseline (NO training): off-the-shelf all-MiniLM-L6-v2  ->  PCA to REDUCED_DIM  ->  KMeans.

Run from src/:   python run_baseline.py

Reads   ../data/processed/chunks_all.jsonl   (made by preprocess.py)
Writes  ../outputs/baseline/
            embeddings_full.npy      (n, 384)  raw model output
            embeddings_reduced.npy   (n, REDUCED_DIM)  what all comparisons use
            clusters.csv             chunk_id, doc_id, split, cluster
            metrics.json             sweep, chosen k, metrics, top terms, doc composition
            k_sweep.png, cluster_map.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cluster import (choose_k, cluster_metrics, cluster_top_terms, composition_by_doc,
                     run_kmeans, sweep_k)
from embed import (BASELINE_MODEL, apply_pca, embed_texts, fit_pca, load_st_model,
                   save_embeddings)
from evaluate import plot_cluster_map, plot_k_sweep, project_2d
from preprocess import load_chunks

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = (BASE_DIR / ".." / "outputs" / "baseline").resolve()

REDUCED_DIM = 64
MAX_SEQ_LEN = 128
K_MAX = 15
SEED = 42


def main(k_override: int | None = None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. data ---------------------------------------------------------------
    chunks = load_chunks("all")
    texts = [c["text"] for c in chunks]
    doc_ids = [c["doc_id"] for c in chunks]
    is_train = np.array([c["split"] == "train" for c in chunks])
    print(f"{len(chunks)} chunks | {len(set(doc_ids))} docs | {is_train.sum()} train chunks")

    # 2. embed --------------------------------------------------------------
    model = load_st_model(BASELINE_MODEL, max_seq_length=MAX_SEQ_LEN)
    full = embed_texts(model, texts)
    save_embeddings(OUT_DIR / "embeddings_full.npy", full)

    # 3. reduce (PCA fitted on train chunks only, then applied to everything) ---
    pca = fit_pca(full[is_train], dim=REDUCED_DIM, seed=SEED)
    emb = apply_pca(pca, full)
    save_embeddings(OUT_DIR / "embeddings_reduced.npy", emb)
    print(f"embeddings: {full.shape} -> {emb.shape} "
          f"(PCA keeps {pca.explained_variance_ratio_.sum():.1%} variance)")

    # 4. choose k and cluster --------------------------------------------------
    sweep = sweep_k(emb, k_min=2, k_max=K_MAX, seed=SEED)
    k = k_override or choose_k(sweep)
    labels = run_kmeans(emb, k, seed=SEED)
    print(f"chosen k = {k} (n_docs = {len(set(doc_ids))})")

    # 5. metrics ------------------------------------------------------------
    metrics = cluster_metrics(emb, labels, true_labels=doc_ids)
    top_terms = cluster_top_terms(texts, labels)
    composition = composition_by_doc(labels, doc_ids)
    print(json.dumps(metrics, indent=2))
    for c, terms in top_terms.items():
        print(f"  cluster {c:>2} ({sum(composition[c].values()):>3} chunks): {', '.join(terms)}")

    # 6. save ---------------------------------------------------------------
    with open(OUT_DIR / "clusters.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "doc_id", "split", "cluster"])
        for c, l in zip(chunks, labels):
            w.writerow([c["chunk_id"], c["doc_id"], c["split"], int(l)])

    result = {
        "model": BASELINE_MODEL, "reduced_dim": int(emb.shape[1]), "k": int(k),
        "metrics": metrics, "sweep": sweep,
        "top_terms": top_terms, "doc_composition": composition,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    plot_k_sweep(sweep, k, "Baseline: k selection", OUT_DIR / "k_sweep.png")
    xy = project_2d(emb, seed=SEED)
    plot_cluster_map(xy, labels, doc_ids, "Baseline", OUT_DIR / "cluster_map.png")
    print(f"saved to {OUT_DIR}")
    return result


if __name__ == "__main__":
    main()
