"""
run_custom.py
-------------
Fine-tune the custom embedder on TRAIN documents, embed all chunks, cluster with the SAME k / algorithm
as the baseline, and print a side-by-side metric comparison. Every experiment variant gets its own
output folder, so runs never overwrite each other.

Run from src/ (after preprocess.py and run_baseline.py):

    python run_custom.py                                   # default variant   -> ../outputs/custom/
    python run_custom.py --tag top2 --top-layers 2         # only top 2 layers -> ../outputs/custom_top2/
    python run_custom.py --tag pfar0 --p-far 0             # nearby-only pairs -> ../outputs/custom_pfar0/
    python run_custom.py --tag top2_pfar0 --top-layers 2 --p-far 0

Or from Python:  run_custom.main(tag="top2", top_layers=2, p_far=0.0)

Options (all optional; defaults in DEFAULTS):
    --top-layers N      fine-tune only the top N of the 6 transformer layers (0 = projection head only)
    --p-far X           prob. that a positive is ANY chunk of the doc instead of a nearby one (0 = nearby only)
    --window W          "nearby" = within +-W chunk positions
    --pairs-per-chunk N positives sampled per anchor chunk
    --no-mask-same-doc  do not mask same-document chunks out of the negatives
    --epochs, --batch-size, --lr-encoder, --lr-proj, --dim, --seed
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from cluster import (choose_k, cluster_metrics, cluster_top_terms, composition_by_doc,
                     run_kmeans, sweep_k)
from embed import embed_texts, load_embeddings, save_embeddings
from evaluate import plot_cluster_map, plot_k_sweep, plot_loss_curve, project_2d
from preprocess import load_chunks

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS = (BASE_DIR / ".." / "outputs").resolve()
BASELINE_DIR = OUTPUTS / "baseline"

DEFAULTS = dict(
    tag=None,              # None -> outputs/custom ; "abc" -> outputs/custom_abc
    dim=64,                # must match the baseline's reduced dim (checked below)
    epochs=3,
    batch_size=64,         # lower to 32 if you hit CUDA OOM on 6GB
    lr_encoder=2e-5,
    lr_proj=1e-4,
    max_seq_len=128,
    top_layers=None,       # None = whole encoder; N = only top N layers; 0 = projection only
    p_far=0.25,
    window=3,
    pairs_per_chunk=2,
    mask_same_doc=True,
    seed=42,
)


def out_dir_for(tag: str | None) -> Path:
    return OUTPUTS / ("custom" if tag in (None, "", "custom") else f"custom_{tag}")


# --------------------------------------------------------------------------- #
# Held-out metrics
# --------------------------------------------------------------------------- #
def subset_silhouette(emb, labels, mask):
    """Silhouette using only held-out (test-document) chunks and the global cluster labels."""
    lab = labels[mask]
    if mask.sum() < 3 or not (1 < len(set(lab.tolist())) < mask.sum()):
        return None
    return float(silhouette_score(emb[mask], lab, metric="cosine"))


def test_doc_metrics(emb, labels, doc_ids, is_test, seed=42):
    """
    ARI / NMI against doc_id using ONLY held-out (test) documents, in two ways:
      *_test_subset    : take the global clustering (k clusters) and score only the test chunks
      *_test_recluster : re-cluster the test chunks alone with k = number of test docs
    Needs >= 2 test documents, otherwise ARI/NMI are undefined -> None values.
    """
    keys = ["ari_test_subset", "nmi_test_subset", "ari_test_recluster", "nmi_test_recluster"]
    ref = np.array(doc_ids)[is_test]
    n_docs = len(set(ref.tolist()))
    if n_docs < 2:
        return {k: None for k in keys}
    sub = labels[is_test]
    re_labels = run_kmeans(emb[is_test], n_docs, seed=seed)
    return {
        "ari_test_subset": float(adjusted_rand_score(ref, sub)),
        "nmi_test_subset": float(normalized_mutual_info_score(ref, sub)),
        "ari_test_recluster": float(adjusted_rand_score(ref, re_labels)),
        "nmi_test_recluster": float(normalized_mutual_info_score(ref, re_labels)),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(**overrides) -> dict:
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown option(s): {sorted(unknown)}; valid: {sorted(DEFAULTS)}")
    cfg = {**DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}

    from train import build_pairs, save_custom, train_embedder   # lazy: needs torch

    out_dir = out_dir_for(cfg["tag"])
    out_dir.mkdir(parents=True, exist_ok=True)
    name = out_dir.name
    print(f"=== variant '{name}' | config: {json.dumps({k: v for k, v in cfg.items() if k != 'tag'})}")

    chunks = load_chunks("all")
    texts = [c["text"] for c in chunks]
    doc_ids = [c["doc_id"] for c in chunks]
    is_test = np.array([c["split"] == "test" for c in chunks])
    train_chunks = [c for c in chunks if c["split"] == "train"]
    test_chunks = [c for c in chunks if c["split"] == "test"]

    # baseline reference (same dim, same k)
    base_meta = json.loads((BASELINE_DIR / "metrics.json").read_text(encoding="utf-8"))
    base_emb = load_embeddings(BASELINE_DIR / "embeddings_reduced.npy")
    assert base_emb.shape[1] == cfg["dim"], \
        f"baseline dim {base_emb.shape[1]} != dim {cfg['dim']}; make them equal"
    k = base_meta["k"]

    # 1. pairs --------------------------------------------------------------
    pair_kw = dict(window=cfg["window"], pairs_per_chunk=cfg["pairs_per_chunk"],
                   p_far=cfg["p_far"], seed=cfg["seed"])
    train_pairs = build_pairs(train_chunks, **pair_kw)
    # val loss needs >= 2 held-out docs: with one doc every negative is masked and the loss is trivially 0
    n_test_docs = len({c["doc_id"] for c in test_chunks})
    val_pairs = build_pairs(test_chunks, **pair_kw) if n_test_docs >= 2 else []
    if n_test_docs < 2:
        print("WARNING: <2 held-out documents -> validation loss disabled (raise TEST_FRACTION in preprocess.py)")
    print(f"pairs: {len(train_pairs)} train | {len(val_pairs)} val")

    # 2. train --------------------------------------------------------------
    enc, hist = train_embedder(
        train_pairs, val_pairs, init_texts=[c["text"] for c in train_chunks],
        dim=cfg["dim"], epochs=cfg["epochs"], batch_size=cfg["batch_size"],
        lr_encoder=cfg["lr_encoder"], lr_proj=cfg["lr_proj"], max_seq_length=cfg["max_seq_len"],
        top_layers=cfg["top_layers"], mask_same_doc=cfg["mask_same_doc"], seed=cfg["seed"])
    save_custom(enc, out_dir / "model")

    # 3. embed --------------------------------------------------------------
    emb = embed_texts(enc, texts)
    save_embeddings(out_dir / "embeddings.npy", emb)

    # 4. cluster: same k + algorithm as baseline; also sweep for the plot ----------
    seed = cfg["seed"]
    sweep = sweep_k(emb, k_min=2, k_max=max(base_meta["sweep"]["k"]), seed=seed)
    labels = run_kmeans(emb, k, seed=seed)
    base_labels = run_kmeans(base_emb, k, seed=seed)

    # 5. metrics ------------------------------------------------------------
    m_custom = cluster_metrics(emb, labels, true_labels=doc_ids)
    m_base = cluster_metrics(base_emb, base_labels, true_labels=doc_ids)
    m_custom["silhouette_test_docs_only"] = subset_silhouette(emb, labels, is_test)
    m_base["silhouette_test_docs_only"] = subset_silhouette(base_emb, base_labels, is_test)
    m_custom.update(test_doc_metrics(emb, labels, doc_ids, is_test, seed))
    m_base.update(test_doc_metrics(base_emb, base_labels, doc_ids, is_test, seed))
    print(f"\nheld-out documents: {sorted({d for d, t in zip(doc_ids, is_test) if t})}")

    print(f"comparison at k={k}, dim={cfg['dim']}  (variant '{name}')")
    print(f"{'metric':<28}{'baseline':>12}{'custom':>12}")
    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
    for key in m_base:
        print(f"{key:<28}{fmt(m_base[key]):>12}{fmt(m_custom[key]):>12}")
    print(f"(custom's own best k by silhouette: {choose_k(sweep)})")

    top_terms = cluster_top_terms(texts, labels)
    composition = composition_by_doc(labels, doc_ids)
    for cl, terms in top_terms.items():
        print(f"  cluster {cl:>2} ({sum(composition[cl].values()):>3} chunks): {', '.join(terms)}")

    # 6. save ---------------------------------------------------------------
    with open(out_dir / "clusters.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "doc_id", "split", "cluster"])
        for ch, l in zip(chunks, labels):
            w.writerow([ch["chunk_id"], ch["doc_id"], ch["split"], int(l)])

    result = {"dim": cfg["dim"], "k": k, "history": hist, "metrics": m_custom, "sweep": sweep,
              "top_terms": top_terms, "doc_composition": composition, "config": cfg}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "comparison.json").write_text(json.dumps(
        {"k": k, "dim": cfg["dim"], "config": cfg, "baseline": m_base, "custom": m_custom}, indent=2),
        encoding="utf-8")

    plot_loss_curve(hist, f"Fine-tuning ({name})", out_dir / "loss_curve.png")
    plot_k_sweep(sweep, choose_k(sweep), f"{name}: k selection", out_dir / "k_sweep.png")
    plot_cluster_map(project_2d(emb, seed=seed), labels, doc_ids, name, out_dir / "cluster_map.png")
    print(f"saved to {out_dir}")
    return result


def parse_args(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Fine-tune and evaluate one custom-embedder variant")
    ap.add_argument("--tag", type=str)
    ap.add_argument("--top-layers", type=int)
    ap.add_argument("--p-far", type=float)
    ap.add_argument("--window", type=int)
    ap.add_argument("--pairs-per-chunk", type=int)
    ap.add_argument("--no-mask-same-doc", dest="mask_same_doc", action="store_const", const=False)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr-encoder", type=float)
    ap.add_argument("--lr-proj", type=float)
    ap.add_argument("--dim", type=int)
    ap.add_argument("--seed", type=int)
    return {k: v for k, v in vars(ap.parse_args(argv)).items() if v is not None}


if __name__ == "__main__":
    main(**parse_args())
