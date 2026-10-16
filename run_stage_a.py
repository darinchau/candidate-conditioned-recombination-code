"""Stage A: candidate-conditioned next-chord decision pilot (task: next-decision scoring,
a subproblem of recomposition; not a mashup experiment).

Usage: .venv-gpu/Scripts/python.exe run_stage_a.py [--smoke]"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SEED_DATA = 20261004
K = 16
L = 8
SEEDS = [0, 1, 2]
LRS = [3e-4, 1e-3]
SPLITS_EVAL = ["dev", "test_id", "test_ood"]
FULL_INV_N = 1000  # examples per split for full-inventory ranking
NEURAL = ["B1", "B2", "B3", "V1", "C1"]
TOK_DROP = 0.25  # pre-run design choice (set before any non-smoke run), same for all neural arms

LOG_LINES: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)
    RES.mkdir(exist_ok=True)
    with open(RES / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(msg: str) -> None:
    (RES / "STATUS.txt").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n", encoding="utf-8")
    log("STATUS " + msg)


# ----------------------------------------------------------------------------- data
def bs_bucket(bs: float) -> int:
    return 5 if bs <= 0 else int(min(4, max(0, round(-math.log2(bs)))))


def ql_bucket(ql: float) -> int:
    return 0 if ql <= 0 else int(min(7, max(0, round(math.log2(ql)) + 3)))


class Data:
    def __init__(self, smoke: bool) -> None:
        pieces = json.loads((HERE / "data" / "pieces.json").read_text(encoding="utf-8"))
        inv = json.loads((HERE / "data" / "inventory.json").read_text(encoding="utf-8"))
        if smoke:
            keep: dict[str, int] = defaultdict(int)
            sel = []
            for p in pieces:
                if keep[p["split"]] < (40 if p["split"] == "train" else 8):
                    keep[p["split"]] += 1
                    sel.append(p)
            pieces = sel
        self.inv = inv
        self.I = len(inv)
        self.key2idx = {(x["fig"], x["mode"]): i for i, x in enumerate(inv)}
        # inventory feature table
        f = np.zeros((self.I, 8), dtype=np.int64)  # deg alt app qual inv size mode root
        pcs = np.zeros((self.I, 12), dtype=np.float32)
        bass = np.zeros(self.I, dtype=np.int64)
        for i, x in enumerate(inv):
            f[i] = [x["deg"], x["alt"] + 2, x["app"], x["qual"], x["inv"], x["size"],
                    int(x["mode"] == "minor"), x["root"]]
            for pc in x["pcs"]:
                pcs[i, pc] = 1.0
            bass[i] = x["bass"]
        self.feat, self.pcs, self.bass = f, pcs, bass
        self.mode = f[:, 6]
        # train vocabulary (fig token): count >= 2 in train else UNK(1); PAD 0
        train_counts: Counter[int] = Counter()
        for p in pieces:
            if p["split"] == "train":
                for c in p["chords"]:
                    train_counts[self.key2idx[(c["fig"], c["mode"])]] += 1
        self.train_counts = train_counts
        self.tok = np.ones(self.I, dtype=np.int64)
        nxt = 2
        for i in sorted(train_counts):
            if train_counts[i] >= 2:
                self.tok[i] = nxt
                nxt += 1
        self.V = nxt
        self.in_train = np.array([train_counts.get(i, 0) > 0 for i in range(self.I)])
        # examples
        ex: dict[str, dict[str, list]] = {s: defaultdict(list) for s in ["train"] + SPLITS_EVAL}
        self.seqs: dict[str, list[list[int]]] = defaultdict(list)
        for p in pieces:
            s = p["split"]
            ids = [self.key2idx[(c["fig"], c["mode"])] for c in p["chords"]]
            self.seqs[s].append(ids)
            ch = p["chords"]
            for t in range(1, len(ch)):
                lo = max(0, t - L)
                ctx = ids[lo:t]
                pad = L - len(ctx)
                d = ex[s]
                d["ctx"].append([-1] * pad + ctx)
                d["cbs"].append([0] * pad + [bs_bucket(c["bs"]) for c in ch[lo:t]])
                d["cql"].append([0] * pad + [ql_bucket(c["ql"]) for c in ch[lo:t]])
                d["ckc"].append([0] * pad + [c["kc"] for c in ch[lo:t]])
                d["qbs"].append(bs_bucket(ch[t]["bs"]))
                d["y"].append(ids[t])
                d["work"].append(p["work"])
        self.ex: dict[str, dict[str, Any]] = {}
        for s, d in ex.items():
            self.ex[s] = {k: (np.array(v) if k != "work" else v) for k, v in d.items()}
            self.ex[s]["unseen"] = ~self.in_train[self.ex[s]["y"]]
        # sampling tables
        self.train_inv = np.where(self.in_train)[0]
        bits = (self.pcs * (1 << np.arange(12))).sum(1).astype(np.int64)
        self.hard: list[np.ndarray] = []
        self.hard_train: list[np.ndarray] = []
        overlap = np.array([[bin(a & b).count("1") for b in bits] for a in bits[:0]])  # placeholder
        del overlap
        pop = np.vectorize(lambda v: bin(int(v)).count("1"))
        for i in range(self.I):
            ov = pop(bits & bits[i])
            m = (self.mode == self.mode[i]) & ((self.feat[:, 7] == self.feat[i, 7]) | (ov >= 2) |
                                                (self.feat[:, 0] == self.feat[i, 0]))
            m[i] = False
            nb = np.where(m)[0]
            self.hard.append(nb)
            self.hard_train.append(nb[self.in_train[nb]])
        cnt = np.array([train_counts.get(i, 0) for i in range(self.I)], dtype=np.float64)
        self.freq_p = {}
        for md in (0, 1):
            w = cnt * (self.mode == md)
            self.freq_p[md] = w / w.sum()

    def candidates(self, split: str, seed: int, train_only: bool) -> np.ndarray:
        """K candidates per example: true + 5 frequency + 5 hard + 5 uniform; true at random slot."""
        y = self.ex[split]["y"]
        n = len(y)
        rng = np.random.default_rng(seed)
        pool = self.train_inv if train_only else np.arange(self.I)
        hard = self.hard_train if train_only else self.hard
        out = np.zeros((n, K), dtype=np.int64)
        modes = self.mode[y]
        fdraw = np.zeros((n, 12), dtype=np.int64)
        for md in (0, 1):
            idx = np.where(modes == md)[0]
            if len(idx):
                fdraw[idx] = rng.choice(self.I, size=(len(idx), 12), p=self.freq_p[md])
        udraw = rng.choice(pool, size=(n, 20))
        for r in range(n):
            t = int(y[r])
            chosen = [t]
            seen = {t}
            for c in fdraw[r]:
                if len(chosen) == 6:
                    break
                if c not in seen:
                    chosen.append(int(c)); seen.add(int(c))
            hb = hard[t]
            if len(hb):
                for c in rng.permutation(hb)[:12]:
                    if len(chosen) == 11:
                        break
                    if c not in seen:
                        chosen.append(int(c)); seen.add(int(c))
            for c in udraw[r]:
                if len(chosen) == K:
                    break
                if c not in seen:
                    chosen.append(int(c)); seen.add(int(c))
            while len(chosen) < K:
                c = int(rng.choice(pool))
                if c not in seen:
                    chosen.append(c); seen.add(c)
            perm = rng.permutation(K)
            out[r] = np.array(chosen)[perm]
        return out


# ----------------------------------------------------------------------------- baselines
class NGram:
    """Interpolated absolute-discounting back-off n-gram (order <= 4) over (figure, mode) tokens.
    Deviation from plan wording: interpolated absolute discounting instead of Katz back-off."""

    def __init__(self, data: Data, order: int = 4, D: float = 0.75) -> None:
        self.I, self.order, self.D = data.I, order, D
        self.tables: list[dict[tuple, Counter]] = [defaultdict(Counter) for _ in range(order)]
        uni = np.ones(self.I)
        for s in data.seqs["train"]:
            seq = [-2] + s
            for t in range(1, len(seq)):
                uni[seq[t]] += 1
                for k in range(1, order):
                    if t - k >= 0:
                        self.tables[k][tuple(seq[t - k:t])][seq[t]] += 1
        self.p1 = uni / uni.sum()
        self.cache: dict[tuple, tuple[np.ndarray, np.ndarray, float, int]] = {}
        for k in range(1, order):
            for h, c in self.tables[k].items():
                idx = np.fromiter(c.keys(), dtype=np.int64)
                val = np.fromiter(c.values(), dtype=np.float64)
                self.cache[h] = (idx, val, float(val.sum()), len(idx))

    def dist(self, ctx: list[int], at_start: bool) -> np.ndarray:
        hist = ([-2] if at_start else []) + ctx
        p = self.p1.copy()
        for k in range(1, self.order):
            if len(hist) < k:
                break
            e = self.cache.get(tuple(hist[-k:]))
            if e is None:
                break  # longer histories cannot exist either
            idx, val, tot, n1 = e
            p *= self.D * n1 / tot
            p[idx] += np.maximum(val - self.D, 0) / tot
        return p

    def score(self, data: Data, split: str, cands: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        e = data.ex[split]
        rows = np.arange(len(e["y"])) if rows is None else rows
        out = np.zeros(cands.shape)
        for j, r in enumerate(rows):
            ctx = [int(c) for c in e["ctx"][r] if c >= 0]
            # context shorter than L only at piece start
            p = self.dist(ctx, at_start=(e["ctx"][r][0] < 0))
            out[j] = np.log(p[cands[j]])
        return out


def heuristic_scores(data: Data, split: str, cands: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """R2: hand-written functional-harmony preferences relative to the previous chord."""
    e = data.ex[split]
    rows = np.arange(len(e["y"])) if rows is None else rows
    prev = e["ctx"][rows, -1]
    f = data.feat
    P = f[prev][:, None, :]
    C = f[cands]
    interval = (C[..., 7] - P[..., 7]) % 12
    s = np.zeros(cands.shape)
    s += 2.0 * (interval == 5) + 0.8 * ((interval == 2) | (interval == 10)) + 0.3 * (interval == 7)
    s -= 1.0 * (interval == 0)
    s += 3.0 * ((P[..., 2] > 0) & (C[..., 0] == P[..., 2]) & (C[..., 2] == 0))
    prev_fig = np.array([data.inv[i]["fig"] for i in prev])
    cad64 = np.array([fg.startswith("Cad") for fg in prev_fig]) | ((P[..., 0, 0] == 1) & (P[..., 0, 4] == 2))
    s += 3.0 * (cad64[:, None] & (C[..., 0] == 5) & (C[..., 2] == 0))
    s += 2.0 * (np.isin(P[..., 0], [5, 7]) & (P[..., 2] == 0) & (C[..., 0] == 1) & (C[..., 2] == 0))
    s += 1.0 * ((C[..., 1] == 2) & (C[..., 2] == 0)) + 0.5 * (C[..., 4] == 0)
    s += 1.0 * (C[..., 6] == P[..., 6])
    return s


# ----------------------------------------------------------------------------- neural arms
class Model(nn.Module):
    def __init__(self, data: Data, arm: str, d: int = 256, layers: int = 4, heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.arm = arm
        self.d = d
        self.tok: torch.Tensor
        self.feat: torch.Tensor
        self.pcs: torch.Tensor
        self.bass: torch.Tensor
        self.register_buffer("tok", torch.tensor(data.tok))
        self.register_buffer("feat", torch.tensor(data.feat))
        self.register_buffer("pcs", torch.tensor(data.pcs))
        self.register_buffer("bass", F.one_hot(torch.tensor(data.bass), 12).float())
        self.e_tok = nn.Embedding(data.V, d)
        sizes = [8, 5, 8, 6, 4, 6, 2, 12]
        self.e_feat = nn.ModuleList(nn.Embedding(n, d) for n in sizes)
        self.e_pcs = nn.Linear(24, d)
        self.e_bs = nn.Embedding(6, d)
        self.e_ql = nn.Embedding(8, d)
        self.e_kc = nn.Embedding(2, d)
        self.e_pos = nn.Embedding(L + 2, d)
        self.e_type = nn.Embedding(3, d)  # 0 ctx, 1 query, 2 candidate
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        if arm == "B1":
            self.head = nn.Linear(d, data.V)
        elif arm == "B2":
            self.heads = nn.ModuleList(nn.Linear(d, n) for n in [8, 5, 8, 6, 4, 2, 12])
            self.head_cols = [0, 1, 2, 3, 4, 6, 7]
        elif arm == "B3":
            self.cand_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            self.proj = nn.Linear(d, d)
        else:
            self.head = nn.Linear(d, 1)

    def chord(self, idx: torch.Tensor) -> torch.Tensor:
        i = idx.clamp(min=0)
        t = self.tok[i]
        if self.training:  # identity-token dropout so features, not the id, must carry the decision
            t = torch.where(torch.rand(t.shape, device=t.device) < TOK_DROP, torch.ones_like(t), t)
        h = self.e_tok(t)
        fe = self.feat[i]
        for j, emb in enumerate(self.e_feat):
            h = h + emb(fe[..., j])
        return h + self.e_pcs(torch.cat([self.pcs[i], self.bass[i]], -1))

    def context(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = b["ctx"]
        pos = torch.arange(L, device=ctx.device)
        h = (self.chord(ctx) + self.e_bs(b["cbs"]) + self.e_ql(b["cql"]) + self.e_kc(b["ckc"])
             + self.e_pos(pos) + self.e_type.weight[0])
        return h, ctx < 0

    def query(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.e_bs(b["qbs"]) + self.e_pos.weight[L] + self.e_type.weight[1]

    def forward(self, b: dict[str, torch.Tensor], cands: torch.Tensor) -> torch.Tensor:
        """Returns candidate scores [B, Kc] (log-domain for B1/B2)."""
        B, Kc = cands.shape
        if self.arm in ("B1", "B2", "B3"):
            h, m = self.context(b)
            q = self.query(b)[:, None]
            x = torch.cat([h, q], 1)
            m = torch.cat([m, torch.zeros(B, 1, dtype=torch.bool, device=m.device)], 1)
            z = self.norm(self.enc(x, src_key_padding_mask=m))[:, -1]
            if self.arm == "B1":
                return F.log_softmax(self.head(z), -1).gather(1, self.tok[cands])
            if self.arm == "B2":
                fe = self.feat[cands]
                s = torch.zeros(B, Kc, device=z.device)
                for hd, col in zip(self.heads, self.head_cols):
                    s = s + F.log_softmax(hd(z), -1).gather(1, fe[..., col])
                return s
            c = self.cand_mlp(self.chord(cands))
            return torch.einsum("bd,bkd->bk", self.proj(z), c) / math.sqrt(self.d)
        # cross-encoder: one sequence per (context, candidate)
        cand = (self.chord(cands) + self.query(b)[:, None] + self.e_pos.weight[L + 1]
                + self.e_type.weight[2])  # [B,Kc,d]
        if self.arm == "C1":
            x = cand.reshape(B * Kc, 1, self.d)
            z = self.norm(self.enc(x))[:, -1]
        else:
            h, m = self.context(b)
            x = torch.cat([h[:, None].expand(B, Kc, L, self.d), cand[:, :, None]], 2).reshape(B * Kc, L + 1, self.d)
            m = torch.cat([m, torch.zeros(B, 1, dtype=torch.bool, device=m.device)], 1)
            m = m[:, None].expand(B, Kc, L + 1).reshape(B * Kc, L + 1)
            z = self.norm(self.enc(x, src_key_padding_mask=m))[:, -1]
        return self.head(z).view(B, Kc)

    def loss(self, b: dict[str, torch.Tensor], cands: torch.Tensor, ypos: torch.Tensor) -> torch.Tensor:
        if self.arm == "B1":
            h, m = self.context(b)
            x = torch.cat([h, self.query(b)[:, None]], 1)
            m = torch.cat([m, torch.zeros(len(m), 1, dtype=torch.bool, device=m.device)], 1)
            z = self.norm(self.enc(x, src_key_padding_mask=m))[:, -1]
            return F.cross_entropy(self.head(z), self.tok[b["y"]])
        if self.arm == "B2":
            h, m = self.context(b)
            x = torch.cat([h, self.query(b)[:, None]], 1)
            m = torch.cat([m, torch.zeros(len(m), 1, dtype=torch.bool, device=m.device)], 1)
            z = self.norm(self.enc(x, src_key_padding_mask=m))[:, -1]
            fe = self.feat[b["y"]]
            return torch.stack([F.cross_entropy(hd(z), fe[:, col])
                                for hd, col in zip(self.heads, self.head_cols)]).sum()
        return F.cross_entropy(self.forward(b, cands), ypos)


def to_batch(e: dict, rows: np.ndarray, dev: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(e[k][rows], device=dev) for k in ("ctx", "cbs", "cql", "ckc", "qbs", "y")}


@torch.no_grad()
def score_neural(model: Model, data: Data, split: str, cands: np.ndarray, dev: torch.device,
                 rows: np.ndarray | None = None, bs: int = 512) -> np.ndarray:
    model.eval()
    e = data.ex[split]
    rows = np.arange(len(e["y"])) if rows is None else rows
    out = np.zeros(cands.shape, dtype=np.float32)
    kc = cands.shape[1]
    step = max(1, bs * K // kc) if model.arm in ("V1", "C1") else bs
    for i in range(0, len(rows), step):
        r = rows[i:i + step]
        b = to_batch(e, r, dev)
        c = torch.as_tensor(cands[i:i + step], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            s = model(b, c)
        out[i:i + step] = s.float().cpu().numpy()
    return out


def train_neural(data: Data, arm: str, seed: int, lr: float, dev: torch.device, epochs: int,
                 train_cands: list[np.ndarray], dev_cands: np.ndarray) -> tuple[Model, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Model(data, arm).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    e = data.ex["train"]
    n = len(e["y"])
    bs = 256
    steps = epochs * math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(seed)
    best: tuple[float, dict[str, torch.Tensor], int] = (float("inf"), {}, -1)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        cands_ep = train_cands[ep % len(train_cands)]
        ypos_all = (cands_ep == e["y"][:, None]).argmax(1)
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            r = perm[i:i + bs]
            b = to_batch(e, r, dev)
            c = torch.as_tensor(cands_ep[r], device=dev)
            yp = torch.as_tensor(ypos_all[r], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = model.loss(b, c, yp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(r)
        s = score_neural(model, data, "dev", dev_cands, dev)
        m = kset_metrics(s, dev_cands, data.ex["dev"]["y"])
        hist.append({"epoch": ep, "train_loss": tot / n, "dev_nll": m["nll"], "dev_top1": m["top1"]})
        if m["nll"] < best[0]:
            best = (m["nll"], {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
        if time.time() - t0 > 20 * 60:
            log(f"  {arm} seed{seed} lr{lr}: 20-min budget hit at epoch {ep}")
            break
    model.load_state_dict(best[1])
    return model, {"history": hist, "best_epoch": best[2], "train_seconds": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------- metrics
def jitter(scores: np.ndarray) -> np.ndarray:
    """Fixed-seed random tie-breaking so tied scores (R1 back-off, R2 rules) are not credited as correct."""
    return np.random.default_rng(12345).random(scores.shape) * 1e-6


def logp(scores: np.ndarray) -> np.ndarray:
    s = scores - scores.max(1, keepdims=True)
    return s - np.log(np.exp(s).sum(1, keepdims=True))


def kset_metrics(scores: np.ndarray, cands: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is not None:
        scores, cands, y = scores[mask], cands[mask], y[mask]
    if len(y) == 0:
        return {"n": 0}
    lp = logp(scores + jitter(scores))
    ypos = (cands == y[:, None]).argmax(1)
    ly = lp[np.arange(len(y)), ypos]
    rank = (lp > ly[:, None]).sum(1)
    p = np.exp(lp)
    conf = p.max(1)
    corr = rank == 0
    bins = np.minimum((conf * 15).astype(int), 14)
    ece = sum(abs(corr[bins == k].mean() - conf[bins == k].mean()) * (bins == k).mean()
              for k in range(15) if (bins == k).any())
    return {"n": int(len(y)), "top1": float(corr.mean()), "top3": float((rank < 3).mean()),
            "nll": float(-ly.mean()), "ece15": float(ece)}


def per_example_correct(scores: np.ndarray, cands: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores = scores + jitter(scores)
    ypos = (cands == y[:, None]).argmax(1)
    ly = scores[np.arange(len(y)), ypos]
    return ((scores > ly[:, None]).sum(1) == 0).astype(np.float64)


def full_inv_metrics(scores: np.ndarray, y: np.ndarray) -> dict:
    scores = scores + jitter(scores)
    ly = scores[np.arange(len(y)), y]
    rank = (scores > ly[:, None]).sum(1)
    return {"n": int(len(y)), "top1": float((rank == 0).mean()), "top5": float((rank < 5).mean()),
            "mrr": float((1.0 / (rank + 1)).mean())}


def cluster_bootstrap(diff: np.ndarray, works: list[str], draws: int = 2000, seed: int = 0) -> dict:
    ws = sorted(set(works))
    wi = {w: i for i, w in enumerate(ws)}
    g = np.array([wi[w] for w in works])
    sums = np.bincount(g, weights=diff, minlength=len(ws))
    cnts = np.bincount(g, minlength=len(ws)).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ws), size=(draws, len(ws)))
    bs = sums[idx].sum(1) / cnts[idx].sum(1)
    return {"mean": float(diff.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "works": len(ws)}


# ----------------------------------------------------------------------------- latency
@torch.no_grad()
def latency(model: Model, data: Data, dev: torch.device, kc: int, reps: int = 200) -> dict:
    model.eval()
    e = data.ex["test_ood"]
    rng = np.random.default_rng(0)
    times = []
    for i in range(reps + 20):
        r = np.array([rng.integers(len(e["y"]))])
        b = to_batch(e, r, dev)
        c = torch.as_tensor(rng.choice(data.I, size=(1, kc), replace=False), device=dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        s = model(b, c)
        torch.softmax(s.float(), -1).cpu()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if i >= 20:
            times.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p99_ms": float(np.percentile(times, 99))}


# ----------------------------------------------------------------------------- main
def env_info(dev: torch.device) -> dict:
    def git(*a: str) -> str:
        try:
            return subprocess.check_output(["git", *a], cwd=HERE, text=True).strip()
        except Exception:  # noqa: BLE001
            return "?"
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
            "platform": platform.platform(), "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", ".")),
            "pieces_sha256": hashlib.sha256((HERE / "data" / "pieces.json").read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256((HERE / "split-manifest.json").read_bytes()).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global RES
    if args.smoke:  # smoke runs never touch the real results or FROZEN.json
        RES = HERE / "results-smoke"
    frozen_path = RES / "FROZEN.json" if args.smoke else HERE / "FROZEN.json"
    RES.mkdir(exist_ok=True)
    (RES / "run.log").write_text("", encoding="utf-8")
    t_start = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 2 if args.smoke else 15
    seeds = [0] if args.smoke else SEEDS
    lrs = [1e-3] if args.smoke else LRS
    env = env_info(dev)
    log(f"env {json.dumps(env)}  smoke={args.smoke}")
    status("loading data")
    data = Data(args.smoke)
    log(f"inventory {data.I}, train vocab {data.V}, examples " +
        ", ".join(f"{s}={len(data.ex[s]['y'])}" for s in data.ex))
    status("sampling candidate sets")
    train_cands = [data.candidates("train", SEED_DATA + 100 + i, train_only=True) for i in range(4)]
    cands = {s: data.candidates(s, SEED_DATA + j, train_only=False) for j, s in enumerate(SPLITS_EVAL)}
    full_rows = {s: np.sort(np.random.default_rng(SEED_DATA + 7 + j).choice(
        len(data.ex[s]["y"]), size=min(FULL_INV_N, len(data.ex[s]["y"])), replace=False))
        for j, s in enumerate(SPLITS_EVAL)}
    inv_all = np.arange(data.I)

    scores: dict[str, dict[str, list[np.ndarray]]] = {s: defaultdict(list) for s in SPLITS_EVAL}
    full_scores: dict[str, dict[str, list[np.ndarray]]] = {s: defaultdict(list) for s in SPLITS_EVAL}

    def score_baselines(split: str) -> None:
        y = data.ex[split]["y"]
        rng = np.random.default_rng(1)
        scores[split]["R0"].append(rng.random(cands[split].shape))
        scores[split]["R1"].append(ngram.score(data, split, cands[split]))
        scores[split]["R2"].append(heuristic_scores(data, split, cands[split]))
        fr = full_rows[split]
        fc = np.broadcast_to(inv_all, (len(fr), data.I))
        full_scores[split]["R0"].append(rng.random(fc.shape))
        full_scores[split]["R1"].append(ngram.score(data, split, np.ascontiguousarray(fc), fr))
        full_scores[split]["R2"].append(heuristic_scores(data, split, np.ascontiguousarray(fc), fr))
        del y

    status("baselines R0/R1/R2 on dev")
    ngram = NGram(data)
    score_baselines("dev")

    # ---- dev-only selection: lr grid on seed 0
    dev_sel: dict[str, dict] = {}
    for arm in NEURAL:
        res = {}
        for lr in lrs:
            status(f"dev selection {arm} lr={lr}")
            model, info = train_neural(data, arm, 0, lr, dev, epochs, train_cands, cands["dev"])
            m = kset_metrics(score_neural(model, data, "dev", cands["dev"], dev), cands["dev"], data.ex["dev"]["y"])
            res[str(lr)] = {"dev": m, **info}
            log(f"  {arm} lr={lr} dev top1={m['top1']:.4f} nll={m['nll']:.4f} best_ep={info['best_epoch']}")
            del model
            torch.cuda.empty_cache()
        best_lr = min(res, key=lambda k: res[k]["dev"]["nll"])
        dev_sel[arm] = {"grid": res, "lr": float(best_lr)}
    dev_base = {a: kset_metrics(scores["dev"][a][0], cands["dev"], data.ex["dev"]["y"]) for a in ("R0", "R1", "R2")}
    (RES / "stageA-dev.json").write_text(json.dumps({"baselines": dev_base, "neural": dev_sel}, indent=1), encoding="utf-8")

    # stop rule and leakage control on dev (seed-0 selected configs)
    r1 = dev_base["R1"]["top1"]
    best_dev = {a: dev_sel[a]["grid"][str(dev_sel[a]["lr"])]["dev"]["top1"] for a in NEURAL}
    stop = not (best_dev["V1"] > r1 or best_dev["B3"] > r1)
    leak = best_dev["C1"] > r1 + 0.05
    comparator = "B2" if best_dev["B2"] >= best_dev["B3"] else "B3"
    frozen = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "env": env, "epochs": epochs, "K": K, "L": L,
              "seeds": seeds, "lr": {a: dev_sel[a]["lr"] for a in NEURAL},
              "comparator_for_H1": comparator, "dev_top1": {**{a: dev_base[a]["top1"] for a in dev_base}, **best_dev},
              "stop_rule_triggered": stop, "leakage_flag_C1_gt_R1_plus_5pp": leak,
              "selection": "lr by dev K-set NLL, seed 0; epoch by dev NLL; comparator = better of B2/B3 on dev top-1"}
    frozen_path.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    log(f"FROZEN written: {json.dumps({k: frozen[k] for k in ('lr', 'comparator_for_H1', 'stop_rule_triggered', 'leakage_flag_C1_gt_R1_plus_5pp')})}")
    if stop and not args.smoke:  # smoke ignores the stop rule to exercise the test path
        status("STOPPED by pre-registered rule: neither V1 nor B3 beats R1 on dev; test not evaluated")
        write_report(None, frozen, dev_base, dev_sel, env, t_start)
        return

    # ---- final: seeds x selected lr, evaluate all splits
    status("baselines on test splits")
    for s in ("test_id", "test_ood"):
        score_baselines(s)
    lat: dict[str, dict] = {}
    train_info: dict[str, list] = defaultdict(list)
    for arm in NEURAL:
        for seed in seeds:
            status(f"final {arm} seed={seed}")
            model, info = train_neural(data, arm, seed, dev_sel[arm]["lr"], dev, epochs, train_cands, cands["dev"])
            train_info[arm].append({"seed": seed, "best_epoch": info["best_epoch"], "train_seconds": info["train_seconds"]})
            for s in SPLITS_EVAL:
                scores[s][arm].append(score_neural(model, data, s, cands[s], dev))
                fr = full_rows[s]
                fc = np.ascontiguousarray(np.broadcast_to(inv_all, (len(fr), data.I)))
                full_scores[s][arm].append(score_neural(model, data, s, fc, dev, rows=fr, bs=8))
            if seed == seeds[0]:
                lat[arm] = {}
                for kc in (16, 64, 255):
                    lat[arm][f"gpu_K{kc}"] = latency(model, data, dev, kc) if dev.type == "cuda" else None
                cpu_model = model.float().cpu()
                for kc in (16, 64, 255):
                    lat[arm][f"cpu_K{kc}"] = latency(cpu_model, data, torch.device("cpu"), kc, reps=50)
            del model
            torch.cuda.empty_cache()

    status("metrics + bootstrap")
    arms = ["R0", "R1", "R2"] + NEURAL
    results: dict = {"kset": {}, "full_inventory": {}, "bootstrap": {}, "latency_ms_batch1": lat,
                     "train_info": train_info, "frozen": frozen, "env": env}
    for s in SPLITS_EVAL:
        e = data.ex[s]
        y, uns = e["y"], e["unseen"]
        results["kset"][s] = {}
        results["full_inventory"][s] = {}
        for a in arms:
            per_seed = [kset_metrics(sc, cands[s], y) for sc in scores[s][a]]
            sub = {"all": per_seed,
                   "true_seen_in_train": [kset_metrics(sc, cands[s], y, ~uns) for sc in scores[s][a]],
                   "true_unseen_in_train": [kset_metrics(sc, cands[s], y, uns) for sc in scores[s][a]]}
            results["kset"][s][a] = {k: summarise(v) for k, v in sub.items()}
            results["full_inventory"][s][a] = summarise([full_inv_metrics(sc, y[full_rows[s]]) for sc in full_scores[s][a]])
        # paired bootstrap of V1 - comparator, averaged over seeds, plus seen/unseen gaps
        cv = np.mean([per_example_correct(sc, cands[s], y) for sc in scores[s]["V1"]], 0)
        cb = np.mean([per_example_correct(sc, cands[s], y) for sc in scores[s][comparator]], 0)
        cr = np.mean([per_example_correct(sc, cands[s], y) for sc in scores[s]["R1"]], 0)
        w = e["work"]
        bt = {f"V1_minus_{comparator}": cluster_bootstrap(cv - cb, w),
              "V1_minus_R1": cluster_bootstrap(cv - cr, w)}
        for name, msk in (("seen", ~uns), ("unseen", uns)):
            if msk.sum() > 0:
                bt[f"V1_minus_{comparator}_{name}"] = cluster_bootstrap((cv - cb)[msk], [w[i] for i in np.where(msk)[0]])
        results["bootstrap"][s] = bt
    o = results["bootstrap"]["test_ood"]
    gap_all = o[f"V1_minus_{comparator}"]
    gap_seen = o.get(f"V1_minus_{comparator}_seen", {}).get("mean")
    gap_unseen = o.get(f"V1_minus_{comparator}_unseen", {}).get("mean")
    h1 = {"criterion": "V1 - comparator top-1 on test_ood: CI lower > 0 AND unseen-true gap >= seen-true gap",
          "ci_lower": gap_all["ci95"][0], "gap_seen": gap_seen, "gap_unseen": gap_unseen,
          "supported": bool(gap_all["ci95"][0] > 0 and gap_unseen is not None and gap_seen is not None
                            and gap_unseen >= gap_seen)}
    results["H1"] = h1
    results["wall_seconds"] = round(time.time() - t_start, 1)
    results["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else None
    (RES / "stageA-results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    write_report(results, frozen, dev_base, dev_sel, env, t_start)
    status(f"DONE in {results['wall_seconds'] / 60:.1f} min; H1 supported={h1['supported']}")


def summarise(ms: list[dict]) -> dict:
    if not ms or ms[0].get("n", 0) == 0:
        return {"n": 0}
    out = {"n": ms[0]["n"], "seeds": len(ms)}
    for k in ms[0]:
        if k == "n":
            continue
        v = np.array([m[k] for m in ms])
        out[k] = float(v.mean())
        out[k + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return out


def write_report(results: dict | None, frozen: dict, dev_base: dict, dev_sel: dict, env: dict, t0: float) -> None:
    L_ = ["# Stage A report: candidate-conditioned next-chord decision (preliminary pilot)", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by run_stage_a.py. Commit {env['commit']} (dirty={env['dirty']}).",
          f"Device {env['device']}, torch {env['torch']}. Wall {(time.time() - t0) / 60:.1f} min.", "",
          "Task: next-decision scoring over K=16 candidates (true + 5 frequency + 5 hard + 5 uniform). "
          "Corpus: When-in-Rome Roman-numeral analyses (CC BY-SA 4.0) at 1c61fe41. "
          "test_ood = Piano_Sonatas + Keyboard_Other (collection-disjoint).", "",
          "## Dev selection (seed 0)", "", "| arm | lr | dev top-1 | dev NLL |", "|---|---|---|---|"]
    for a, m in dev_base.items():
        L_.append(f"| {a} | - | {m['top1']:.3f} | {m['nll']:.3f} |")
    for a, v in dev_sel.items():
        m = v["grid"][str(v["lr"])]["dev"]
        L_.append(f"| {a} | {v['lr']} | {m['top1']:.3f} | {m['nll']:.3f} |")
    L_ += ["", f"Stop rule triggered: {frozen['stop_rule_triggered']}. "
           f"C1 leakage flag: {frozen['leakage_flag_C1_gt_R1_plus_5pp']}. Comparator: {frozen['comparator_for_H1']}.", ""]
    if results is not None:
        for s in SPLITS_EVAL:
            L_ += [f"## {s} (mean ± sd over seeds)", "",
                   "| arm | top-1 | top-3 | NLL | ECE | top-1 true-unseen | full-inv top-1 | full-inv MRR |",
                   "|---|---|---|---|---|---|---|---|"]
            for a, v in results["kset"][s].items():
                al, un, fi = v["all"], v["true_unseen_in_train"], results["full_inventory"][s][a]
                uns = f"{un['top1']:.3f} (n={un['n']})" if un.get("n") else "-"
                L_.append(f"| {a} | {al['top1']:.3f}±{al['top1_sd']:.3f} | {al['top3']:.3f} | {al['nll']:.3f} | "
                          f"{al['ece15']:.3f} | {uns} | {fi['top1']:.3f} | {fi['mrr']:.3f} |")
            L_ += ["", "Work-cluster bootstrap (2000 draws, seed-averaged correctness):", ""]
            for k, b in results["bootstrap"][s].items():
                L_.append(f"- {k}: {b['mean']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] over {b['works']} works")
            L_.append("")
        h = results["H1"]
        L_ += ["## H1", "", f"{h['criterion']}.", "",
               f"CI lower {h['ci_lower']:+.4f}; seen gap {h['gap_seen']}; unseen gap {h['gap_unseen']}. "
               f"Supported: {h['supported']}.", "", "## Latency, batch 1 (ms p50/p99)", ""]
        for a, v in results["latency_ms_batch1"].items():
            L_.append(f"- {a}: " + "; ".join(f"{k} {x['p50_ms']:.2f}/{x['p99_ms']:.2f}" for k, x in v.items() if x))
        L_ += ["", f"Peak GPU memory {results['peak_gpu_mem_gb']} GB.", ""]
    L_ += ["Caveats: preliminary single-corpus pilot; analyses are expert annotations, not audio or listener "
           "judgements; next-chord accuracy is a proxy and says nothing about musical quality."]
    (RES / "stageA-report.md").write_text("\n".join(L_) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:  # noqa: BLE001
        import traceback
        status(f"FAILED: {type(ex).__name__}: {ex}")
        log(traceback.format_exc())
        raise
