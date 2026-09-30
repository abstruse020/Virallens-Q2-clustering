"""
main.py
-------
Entry point for Q2: domain-specific embedder vs baseline.

    python main.py                                       # run whatever is missing, print comparison
    python main.py --tag top2 --top-layers 2             # new variant -> outputs/custom_top2/
    python main.py --tag pfar0 --p-far 0                 # nearby-only positives
    python main.py --tag top2_pfar0 --top-layers 2 --p-far 0
    python main.py --stages                              # (no stages) just print the comparison of all variants
    python main.py --compare custom top2 pfar0           # compare only these variants

    --force-preprocess   re-parse PDFs (also re-runs baseline + custom)
    --force-baseline     re-run baseline only
    --force-custom       re-train this variant even if its results exist

Pipeline
    1. preprocess : ../data/*.pdf -> cleaned, chunked, doc-level train/test split   (preprocess.py)
    2. baseline   : pretrained all-MiniLM-L6-v2 -> PCA -> KMeans + metrics         (run_baseline.py)
    3. custom     : fine-tuned encoder + projection -> KMeans + metrics            (run_custom.py)
    4. report     : baseline + every variant side by side                          (outputs/comparison.md)

Stages are skipped when their outputs already exist, unless forced.
Works if placed in src/ (next to the other modules) or one level above it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
if SRC.is_dir():                      # main.py at project root, modules in src/
    sys.path.insert(0, str(SRC))
    ROOT = HERE
else:                                 # main.py inside src/
    sys.path.insert(0, str(HERE))
    ROOT = HERE.parent

PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
BASELINE_DIR = OUTPUTS / "baseline"
REPORT_PATH = OUTPUTS / "comparison.md"

STAGES = ("preprocess", "baseline", "custom")

# metric -> (label, higher_is_better)
METRICS = {
    "silhouette_cosine": ("Silhouette (cosine), all", True),
    "davies_bouldin": ("Davies-Bouldin, all", False),
    "calinski_harabasz": ("Calinski-Harabasz, all", True),
    "ari_vs_reference": ("ARI vs doc_id, all", True),
    "nmi_vs_reference": ("NMI vs doc_id, all", True),
    "silhouette_test_docs_only": ("Silhouette, test docs only", True),
    "ari_test_subset": ("ARI, test docs (global clusters)", True),
    "nmi_test_subset": ("NMI, test docs (global clusters)", True),
    "ari_test_recluster": ("ARI, test docs (re-clustered)", True),
    "nmi_test_recluster": ("NMI, test docs (re-clustered)", True),
}

NOTES = ("Notes: * = best in row. 'all' rows include documents the custom models trained on "
         "(optimistic for them); 'test docs' rows use held-out documents only. "
         "Davies-Bouldin: lower is better.")


def custom_dir(tag: str | None) -> Path:
    return OUTPUTS / ("custom" if tag in (None, "", "custom") else f"custom_{tag}")


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_preprocess(force: bool) -> bool:
    """Returns True if preprocessing actually ran."""
    if not force and (PROCESSED / "chunks_all.jsonl").exists():
        print("[preprocess] found existing chunks -> skipping (use --force-preprocess to redo)")
        return False
    from preprocess import run_preprocess
    print("[preprocess] extracting, cleaning, chunking, splitting ...")
    stats = run_preprocess(use_cache=not force)
    print(f"[preprocess] {stats['n_docs']} docs | {stats['n_chunks']} chunks | "
          f"test docs: {stats['test_docs']} | skipped: {stats['skipped']}")
    return True


def stage_baseline(force: bool) -> None:
    if not force and (BASELINE_DIR / "metrics.json").exists():
        print("[baseline] found existing results -> skipping (use --force-baseline to redo)")
        return
    import run_baseline
    print("[baseline] embedding with pretrained model and clustering ...")
    run_baseline.main()


def stage_custom(force: bool, overrides: dict) -> None:
    d = custom_dir(overrides.get("tag"))
    if not force and (d / "comparison.json").exists():
        print(f"[custom] '{d.name}' already exists -> skipping (use --force-custom to re-train)")
        return
    import run_custom
    print(f"[custom] fine-tuning variant '{d.name}' and clustering ...")
    run_custom.main(**overrides)


# --------------------------------------------------------------------------- #
# Report: baseline + any number of custom variants
# --------------------------------------------------------------------------- #
def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def _discover(compare: list[str] | None) -> dict[str, dict]:
    """name -> comparison.json contents, for the requested (or all existing) custom variants."""
    if compare:
        dirs = [custom_dir(t) for t in compare]
    else:
        dirs = sorted(d for d in OUTPUTS.glob("custom*") if d.is_dir())
    found = {}
    for d in dirs:
        p = d / "comparison.json"
        if p.exists():
            found[d.name.replace("custom_", "", 1) if d.name != "custom" else "custom"] = \
                json.loads(p.read_text(encoding="utf-8"))
        else:
            print(f"[report] no results in {d} -> skipped")
    return found


def _table(variants: dict[str, dict]):
    """-> (columns, rows[(label, [cells])], wins{column: n_best_rows})"""
    first = next(iter(variants.values()))
    columns = ["baseline"] + list(variants)
    wins = {c: 0 for c in columns}
    rows = []
    for key, (label, higher) in METRICS.items():
        vals = [first["baseline"].get(key)] + [v["custom"].get(key) for v in variants.values()]
        present = [v for v in vals if v is not None]
        best = None
        if len(present) >= 2 and max(present) != min(present):
            best = max(present) if higher else min(present)
        cells = []
        for c, v in zip(columns, vals):
            is_best = best is not None and v is not None and v == best
            wins[c] += is_best
            cells.append(_fmt(v) + ("*" if is_best else " "))
        if any(v is not None for v in vals):
            rows.append((label, cells))
    return columns, rows, wins


def _config_lines(variants: dict[str, dict]) -> list[str]:
    out = []
    for name, comp in variants.items():
        cfg = comp.get("config")
        if cfg:
            keep = {k: cfg[k] for k in ("top_layers", "p_far", "window", "mask_same_doc", "epochs",
                                        "lr_encoder") if k in cfg}
            out.append(f"  {name}: {keep}")
    return out


def build_console_table(variants: dict[str, dict]) -> str:
    columns, rows, wins = _table(variants)
    first = next(iter(variants.values()))
    w0 = max(len("Metric"), *(len(r[0]) for r in rows))
    wc = [max(len(c), *(len(r[1][i]) for r in rows)) for i, c in enumerate(columns)]

    def line(label, cells):
        return f"{label:<{w0}}  " + "  ".join(f"{c:>{w}}" for c, w in zip(cells, wc))

    rule = "-" * (w0 + 2 + sum(wc) + 2 * (len(wc) - 1))
    out = [f"Baseline vs custom variants (k={first['k']}, dim={first['dim']})", rule,
           line("Metric", columns), rule, *[line(l, c) for l, c in rows], rule,
           "Rows won: " + ", ".join(f"{c}={wins[c]}" for c in columns), ""]
    cfg = _config_lines(variants)
    if cfg:
        out += ["Variant settings:", *cfg, ""]
    out.append(NOTES)
    return "\n".join(out)


def build_markdown(variants: dict[str, dict]) -> str:
    columns, rows, wins = _table(variants)
    first = next(iter(variants.values()))
    lines = [f"# Baseline vs custom variants (k={first['k']}, dim={first['dim']})", "",
             "| Metric | " + " | ".join(columns) + " |",
             "|---|" + "---:|" * len(columns)]
    lines += [f"| {l} | " + " | ".join(c.strip() for c in cells) + " |" for l, cells in rows]
    lines += ["", "Rows won: " + ", ".join(f"{c}={wins[c]}" for c in columns), "", NOTES]
    cfg = _config_lines(variants)
    if cfg:
        lines += ["", "Variant settings:", "", *[f"- {c.strip()}" for c in cfg]]
    return "\n".join(lines)


def report(compare: list[str] | None = None) -> None:
    variants = _discover(compare)
    if not variants:
        print("[report] no custom results yet -> run the 'custom' stage first")
        return
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_markdown(variants), encoding="utf-8")
    print("\n" + build_console_table(variants))
    print(f"\n[report] markdown version saved to {REPORT_PATH}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Q2: custom domain embedder vs baseline")
    ap.add_argument("--stages", nargs="*", choices=STAGES, default=list(STAGES),
                    help="stages to run (default: all; pass --stages with nothing to only print the report)")
    ap.add_argument("--force-preprocess", action="store_true", help="re-parse PDFs (forces downstream stages)")
    ap.add_argument("--force-baseline", action="store_true")
    ap.add_argument("--force-custom", action="store_true")
    ap.add_argument("--compare", nargs="+", metavar="TAG",
                    help="variants to show in the report (default: every outputs/custom* folder)")
    # custom-variant options (forwarded to run_custom.main)
    ap.add_argument("--tag", type=str, help="name of this variant -> outputs/custom_<tag>/")
    ap.add_argument("--top-layers", type=int, help="fine-tune only the top N transformer layers")
    ap.add_argument("--p-far", type=float, help="prob. of a document-wide positive (0 = nearby only)")
    ap.add_argument("--window", type=int)
    ap.add_argument("--pairs-per-chunk", type=int)
    ap.add_argument("--no-mask-same-doc", dest="mask_same_doc", action="store_const", const=False)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr-encoder", type=float)
    ap.add_argument("--lr-proj", type=float)
    ap.add_argument("--dim", type=int)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args(argv)
    custom_keys = ("tag", "top_layers", "p_far", "window", "pairs_per_chunk", "mask_same_doc",
                   "epochs", "batch_size", "lr_encoder", "lr_proj", "dim", "seed")
    overrides = {k: getattr(args, k) for k in custom_keys if getattr(args, k) is not None}
    return args, overrides


def main(argv=None) -> None:
    args, overrides = parse_args(argv)

    redo_downstream = False
    if "preprocess" in args.stages:
        redo_downstream = stage_preprocess(args.force_preprocess)
    if "baseline" in args.stages:
        stage_baseline(args.force_baseline or redo_downstream)
    if "custom" in args.stages:
        stage_custom(args.force_custom or redo_downstream, overrides)

    report(args.compare)


if __name__ == "__main__":
    main()
