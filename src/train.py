"""
train.py
--------
Domain-specific embedder = pretrained MiniLM encoder + linear projection head (384 -> dim),
fine-tuned with a contrastive (InfoNCE) loss on self-supervised pairs from the TRAIN documents.

Key ideas
  * Positive pair  = two chunks of the same document (mostly nearby chunks, sometimes far ones).
  * Negatives      = the other pairs' chunks in the batch ...
  * ... EXCEPT chunks from the same document, which are masked out (they would be false negatives;
    with few documents this matters a lot).
  * Projection head is initialised from PCA of the pretrained embeddings, so training starts from
    exactly the "baseline reduced to dim" model. Any gain is therefore due to training.

Only depends on torch + sentence-transformers' `tokenize` / forward, so it is robust to
sentence-transformers version changes.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from embed import BASELINE_MODEL, load_st_model


# --------------------------------------------------------------------------- #
# 1. Pair construction
# --------------------------------------------------------------------------- #
def build_pairs(chunks: list[dict], window: int = 3, pairs_per_chunk: int = 2,
                p_far: float = 0.25, seed: int = 42) -> list[tuple[str, str, str]]:
    """
    For every chunk (anchor) sample `pairs_per_chunk` positives from the same document:
      - with prob (1 - p_far): a chunk within +-`window` positions (local context)
      - with prob p_far      : any other chunk of the document (document-level topic)
    Returns a shuffled list of (anchor_text, positive_text, doc_id).
    """
    rng = random.Random(seed)
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)

    pairs = []
    for doc_id, cs in by_doc.items():
        cs = sorted(cs, key=lambda c: c["chunk_idx"])
        n = len(cs)
        if n < 2:
            continue
        for i, a in enumerate(cs):
            for _ in range(pairs_per_chunk):
                if rng.random() < p_far:
                    cand = [j for j in range(n) if j != i]
                else:
                    cand = [j for j in range(max(0, i - window), min(n, i + window + 1)) if j != i]
                pairs.append((a["text"], cs[rng.choice(cand)]["text"], doc_id))
    rng.shuffle(pairs)
    return pairs


# --------------------------------------------------------------------------- #
# 2. Model: SentenceTransformer + projection head
# --------------------------------------------------------------------------- #
class ProjectedEncoder(nn.Module):
    """Encoder -> Linear(dim) -> L2 normalize. Exposes .encode() like a SentenceTransformer."""

    def __init__(self, st_model, dim: int = 64):
        super().__init__()
        self.st = st_model
        self.dim = dim
        self.proj = nn.Linear(st_model.get_sentence_embedding_dimension(), dim)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _base(self, texts: list[str]) -> torch.Tensor:
        """Encoder output before the projection head."""
        feats = self.st.tokenize(texts)
        feats = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in feats.items()}
        return self.st(feats)["sentence_embedding"]

    def forward_texts(self, texts: list[str]) -> torch.Tensor:
        return F.normalize(self.proj(self._base(texts)).float(), dim=-1)

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 64, convert_to_numpy: bool = True,
               normalize_embeddings: bool = True, show_progress_bar: bool = False) -> np.ndarray:
        self.eval()
        starts = range(0, len(texts), batch_size)
        if show_progress_bar:
            from tqdm import tqdm
            starts = tqdm(starts, desc="encode")
        out = []
        use_amp = self.device.type == "cuda"
        for i in starts:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                out.append(self.forward_texts(texts[i:i + batch_size]).float().cpu())
        return torch.cat(out).numpy()


@torch.no_grad()
def init_projection_from_pca(enc: ProjectedEncoder, texts: list[str], batch_size: int = 64,
                             seed: int = 42) -> float:
    """Set proj.weight/bias so that proj(x) == PCA(x). Returns explained variance ratio."""
    enc.eval()
    feats = []
    for i in range(0, len(texts), batch_size):
        feats.append(enc._base(texts[i:i + batch_size]).float().cpu())
    X = torch.cat(feats).numpy()
    dim = min(enc.dim, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=dim, random_state=seed).fit(X)
    W = torch.zeros(enc.dim, X.shape[1])
    W[:dim] = torch.tensor(pca.components_, dtype=torch.float32)
    b = -(W @ torch.tensor(pca.mean_, dtype=torch.float32))
    enc.proj.weight.copy_(W.to(enc.device))
    enc.proj.bias.copy_(b.to(enc.device))
    return float(pca.explained_variance_ratio_.sum())


# --------------------------------------------------------------------------- #
# 3. Loss
# --------------------------------------------------------------------------- #
def info_nce(a: torch.Tensor, p: torch.Tensor, doc_ids: list[str],
             scale: float = 20.0, mask_same_doc: bool = True) -> torch.Tensor:
    """Symmetric InfoNCE. Row i's positive is column i; same-document off-diagonal entries are masked."""
    logits = (a.float() @ p.float().T) * scale
    if mask_same_doc:
        codes = {d: i for i, d in enumerate(dict.fromkeys(doc_ids))}
        d = torch.tensor([codes[x] for x in doc_ids], device=logits.device)
        same = d[:, None] == d[None, :]
        eye = torch.eye(len(d), dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(same & ~eye, float("-inf"))
    target = torch.arange(len(doc_ids), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def _batches(pairs, batch_size, shuffle, rng):
    idx = list(range(len(pairs)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        b = [pairs[j] for j in idx[i:i + batch_size]]
        if len(b) >= 8:  # tiny last batches give a meaningless loss
            yield [x[0] for x in b], [x[1] for x in b], [x[2] for x in b]


@torch.no_grad()
def eval_loss(enc: ProjectedEncoder, pairs, batch_size: int, scale: float,
              mask_same_doc: bool = True) -> float | None:
    if not pairs:
        return None
    enc.eval()
    losses = []
    for A, P, D in _batches(pairs, batch_size, False, None):
        losses.append(info_nce(enc.forward_texts(A), enc.forward_texts(P), D, scale, mask_same_doc).item())
    return float(np.mean(losses)) if losses else None


# --------------------------------------------------------------------------- #
# 4. Partial fine-tuning helpers
# --------------------------------------------------------------------------- #
def _bert_parts(enc: ProjectedEncoder):
    try:
        am = enc.st[0].auto_model
        return am, list(am.encoder.layer)
    except (AttributeError, IndexError, TypeError) as e:
        raise RuntimeError("top_layers needs a BERT-style encoder (st[0].auto_model.encoder.layer)") from e


def set_trainable_layers(enc: ProjectedEncoder, top_layers: int | None) -> None:
    """None -> whole encoder trainable. n -> only the top n transformer layers (0 -> encoder frozen)."""
    if top_layers is None:
        return
    _, layers = _bert_parts(enc)
    n = len(layers)
    top = max(0, min(int(top_layers), n))
    for p in enc.st.parameters():
        p.requires_grad = False
    for layer in layers[n - top:]:
        for p in layer.parameters():
            p.requires_grad = True


def set_train_mode(enc: ProjectedEncoder, top_layers: int | None) -> None:
    """train() mode, but frozen layers stay in eval() so their dropout doesn't add noise."""
    enc.train()
    if top_layers is None:
        return
    am, layers = _bert_parts(enc)
    n = len(layers)
    top = max(0, min(int(top_layers), n))
    am.embeddings.eval()
    for layer in layers[:n - top]:
        layer.eval()


# --------------------------------------------------------------------------- #
# 5. Training
# --------------------------------------------------------------------------- #
def train_embedder(train_pairs, val_pairs=None, init_texts: list[str] | None = None,
                   base_model: str = BASELINE_MODEL, dim: int = 64, epochs: int = 3,
                   batch_size: int = 64, lr_encoder: float = 2e-5, lr_proj: float = 1e-4,
                   scale: float = 20.0, warmup_frac: float = 0.1, max_seq_length: int = 128,
                   top_layers: int | None = None, mask_same_doc: bool = True,
                   seed: int = 42, device: str | None = None):
    """
    top_layers: None = fine-tune the whole encoder; n = only the top n of the 6 transformer layers
                (0 = encoder frozen, only the projection head trains).
    Returns (ProjectedEncoder, history dict).
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = random.Random(seed)

    st = load_st_model(base_model, max_seq_length=max_seq_length, device=device)
    enc = ProjectedEncoder(st, dim).to(st.device)
    ev = init_projection_from_pca(enc, init_texts or [p[0] for p in train_pairs], seed=seed)
    print(f"projection initialised from PCA ({ev:.1%} variance kept)")

    set_trainable_layers(enc, top_layers)
    n_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in enc.parameters())
    print(f"trainable params: {n_train:,} / {n_all:,} (top_layers={top_layers}, mask_same_doc={mask_same_doc})")

    use_amp = enc.device.type == "cuda"
    enc_params = [p for p in enc.st.parameters() if p.requires_grad]
    groups = [{"params": enc_params, "lr": lr_encoder}] if enc_params else []
    groups.append({"params": list(enc.proj.parameters()), "lr": lr_proj})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)

    steps_per_epoch = max(1, len(train_pairs) // batch_size)
    total = steps_per_epoch * epochs
    warm = max(1, int(total * warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist = {"epoch": [0], "train_loss": [None],
            "val_loss": [eval_loss(enc, val_pairs, batch_size, scale, mask_same_doc)]}
    print(f"epoch 0 | val_loss {hist['val_loss'][0]}")

    for ep in range(1, epochs + 1):
        set_train_mode(enc, top_layers)
        t0, losses = time.time(), []
        for A, P, D in _batches(train_pairs, batch_size, True, rng):
            with torch.autocast(device_type=enc.device.type, dtype=torch.float16, enabled=use_amp):
                a, p = enc.forward_texts(A), enc.forward_texts(P)
            loss = info_nce(a, p, D, scale, mask_same_doc)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= scale_before:   # optimizer step was not skipped (no fp16 overflow)
                sched.step()
            losses.append(loss.item())
        tl = float(np.mean(losses)) if losses else None
        vl = eval_loss(enc, val_pairs, batch_size, scale, mask_same_doc)
        hist["epoch"].append(ep); hist["train_loss"].append(tl); hist["val_loss"].append(vl)
        print(f"epoch {ep} | train_loss {tl:.4f} | val_loss {vl if vl is None else round(vl, 4)} "
              f"| {time.time() - t0:.0f}s")
    return enc, hist


# --------------------------------------------------------------------------- #
# 6. Save / load
# --------------------------------------------------------------------------- #
def save_custom(enc: ProjectedEncoder, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    enc.st.save(str(out_dir / "st"))
    torch.save(enc.proj.state_dict(), out_dir / "proj.pt")
    (out_dir / "config.json").write_text(json.dumps(
        {"dim": enc.dim, "max_seq_length": enc.st.max_seq_length}), encoding="utf-8")


def load_custom(out_dir: Path, device: str | None = None) -> ProjectedEncoder:
    cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    st = load_st_model(str(out_dir / "st"), max_seq_length=cfg["max_seq_length"], device=device)
    enc = ProjectedEncoder(st, cfg["dim"]).to(st.device)
    enc.proj.load_state_dict(torch.load(out_dir / "proj.pt", map_location=enc.device))
    return enc.eval()
