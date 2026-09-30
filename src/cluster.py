"""
cluster.py
----------
Traditional clustering + separation metrics. Works on any (n, d) embedding matrix, so the
exact same code is used for the baseline and the custom models (fair comparison).

Embeddings are expected L2-normalized: Euclidean KMeans then behaves like cosine KMeans.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

SEED = 42


def run_kmeans(emb: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(emb)


def run_agglomerative(emb: np.ndarray, k: int) -> np.ndarray:
    """Cross-check algorithm (average linkage on cosine distance)."""
    return AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(emb)


def sweep_k(emb: np.ndarray, k_min: int = 2, k_max: int = 15, seed: int = SEED) -> dict:
    """Run KMeans for every k and record silhouette (cosine), Davies-Bouldin and inertia."""
    k_max = min(k_max, emb.shape[0] - 1)
    out = {"k": [], "silhouette": [], "davies_bouldin": [], "inertia": []}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(emb)
        out["k"].append(k)
        out["silhouette"].append(float(silhouette_score(emb, km.labels_, metric="cosine")))
        out["davies_bouldin"].append(float(davies_bouldin_score(emb, km.labels_)))
        out["inertia"].append(float(km.inertia_))
    return out


def choose_k(sweep: dict) -> int:
    """k with the highest silhouette score."""
    return int(sweep["k"][int(np.argmax(sweep["silhouette"]))])


def cluster_metrics(emb: np.ndarray, labels: np.ndarray, true_labels=None) -> dict:
    """Internal separation metrics, plus ARI/NMI if reference labels (e.g. doc_id) are given."""
    m = {
        "n_clusters": int(len(set(labels))),
        "silhouette_cosine": float(silhouette_score(emb, labels, metric="cosine")),
        "davies_bouldin": float(davies_bouldin_score(emb, labels)),          # lower is better
        "calinski_harabasz": float(calinski_harabasz_score(emb, labels)),    # higher is better
    }
    if true_labels is not None:
        m["ari_vs_reference"] = float(adjusted_rand_score(true_labels, labels))
        m["nmi_vs_reference"] = float(normalized_mutual_info_score(true_labels, labels))
    return m


def cluster_top_terms(texts: list[str], labels: np.ndarray, n_terms: int = 8) -> dict[int, list[str]]:
    """Most distinctive TF-IDF terms per cluster. A human-readable sanity check of what each cluster is about."""
    vec = TfidfVectorizer(stop_words="english", max_df=0.7, min_df=2, sublinear_tf=True)
    try:
        X = vec.fit_transform(texts)
    except ValueError:  # tiny corpus: relax min_df
        vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    labels = np.asarray(labels)
    top = {}
    for c in sorted(set(labels.tolist())):
        centroid = np.asarray(X[labels == c].mean(axis=0)).ravel()
        top[int(c)] = terms[np.argsort(centroid)[::-1][:n_terms]].tolist()
    return top


def composition_by_doc(labels: np.ndarray, doc_ids: list[str]) -> dict[int, dict[str, int]]:
    """For each cluster, how many chunks come from each document."""
    comp: dict[int, dict[str, int]] = {}
    for c, d in zip(labels.tolist(), doc_ids):
        comp.setdefault(int(c), {})
        comp[int(c)][d] = comp[int(c)].get(d, 0) + 1
    return {c: dict(sorted(v.items(), key=lambda kv: -kv[1])) for c, v in sorted(comp.items())}
