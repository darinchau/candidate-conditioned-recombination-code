"""Stage B step 1: candidate-conditioned next beat-block decision (Bach chorales -> Lieder OOD),
implementing the frozen plan-stageB.md.

Writes:
  results-b/STATUS.txt, run.log          progress (read STATUS first)
  results-b/stageB-dev.json              dev-only selection, gates
  FROZEN-B.json                          selected lrs + gate outcomes, written BEFORE any test scoring
  results-b/stageB-results.json          full metrics, bootstrap CIs, env
  results-b/stageB-report.md             human-readable tables + HB1-HB3 verdicts
  models-b/<arm>-seed<k>.pt              final weights (git-ignored)

Usage: .venv-gpu/Scripts/python.exe run_stage_b.py [--smoke]
"""
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

from run_stage_a import cluster_bootstrap, kset_metrics, logp, jitter, per_example_correct, summarise

HERE = Path(__file__).resolve().parent
RES = HERE / "results-b"
MODELS = HERE / "models-b"
SEED_DATA = 20261004
K = 16
L = 8
SEEDS = [0, 1, 2]
LRS = [3e-4, 1e-3]
SPLITS_EVAL = ["dev", "test_id", "test_ood"]
NEURAL = ["B3", "V1", "C1"]
TOK_DROP = 0.25
NV = 79  # cell value codes: 0 PAD, 1 REST, 2.. = MIDI 24..100
NEG_TYPES = ["true", "freq", "hard", "unif"]

LOG_LINES: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    RES.mkdir(exist_ok=True)
    with open(RES / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(msg: str) -> None:
    RES.mkdir(exist_ok=True)
    (RES / "STATUS.txt").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n", encoding="utf-8")
    log("STATUS " + msg)


# ----------------------------------------------------------------------------- data
def code(v: int) -> int:
    return v if v < 2 else v - 24 + 2


class Data:
    def __init__(self, path: Path, splits_eval: list[str] | None = None) -> None:
        self.path = path
        se = SPLITS_EVAL if splits_eval is None else splits_eval
        pieces = json.loads(path.read_text(encoding="utf-8"))
        self.pieces = pieces
        # block table
        bid2i: dict[str, int] = {}
        vals: list[list[int]] = []
        ons: list[list[int]] = []
        for p in pieces:
            for b, v, o in zip(p["bids"], p["vals"], p["ons"]):
                if b not in bid2i:
                    bid2i[b] = len(vals)
                    vals.append(v)
                    ons.append(o)
        self.N = len(vals)
        self.bids = list(bid2i)  # index -> block id hash (step 2 identity checks)
        self.vals = np.array(vals, dtype=np.int64)  # raw: 0 PAD, 1 REST, MIDI
        self.vcode = np.vectorize(code)(self.vals).astype(np.int64)
        self.ons = np.array(ons, dtype=np.int64)
        # slot-0 sonority: 12-hot pc set + bass pc one-hot (lowest sounding at slot 0)
        s0 = self.vals[:, [0, 4, 8, 12]]  # S, A, T, B at slot 0
        self.pcs = np.zeros((self.N, 24), dtype=np.float32)
        self.sop_pc = np.full(self.N, -1, dtype=np.int64)
        self.bass_pc = np.full(self.N, -1, dtype=np.int64)
        for i in range(self.N):
            snd = [int(x) for x in s0[i] if x >= 24]
            for x in snd:
                self.pcs[i, x % 12] = 1.0
            if snd:
                self.pcs[i, 12 + min(snd) % 12] = 1.0
                self.bass_pc[i] = min(snd) % 12
            if s0[i, 0] >= 24:
                self.sop_pc[i] = int(s0[i, 0]) % 12
        # vocabulary: block-id token if train instance count >= 2, else UNK(1); PAD 0
        tc: Counter[int] = Counter()
        tgt: Counter[int] = Counter()
        for p in pieces:
            if p["split"] == "train":
                ids = [bid2i[b] for b in p["bids"]]
                tc.update(ids)
                tgt.update(ids[1:])
        self.train_counts = tc
        self.tok = np.ones(self.N, dtype=np.int64)
        nxt = 2
        for i in sorted(tc):
            if tc[i] >= 2:
                self.tok[i] = nxt
                nxt += 1
        self.V = nxt
        self.in_train_target = np.array([tgt.get(i, 0) > 0 for i in range(self.N)])
        # examples + per-split instance pools
        ex: dict[str, dict[str, list]] = {s: defaultdict(list) for s in ["train"] + se}
        pool: dict[str, dict[str, list]] = {s: defaultdict(list) for s in ["train"] + se}
        self.seqs: dict[str, list[list[int]]] = defaultdict(list)
        for p in pieces:
            s = p["split"]
            ids = [bid2i[b] for b in p["bids"]]
            met = [[m[0], m[1], m[2], p["mode"]] for m in p["beats"]]  # bpb, bi, comp, mode
            self.seqs[s].append(ids)
            for i, m in zip(ids, met):
                pool[s]["bid"].append(i)
                pool[s]["work"].append(p["work"])
                pool[s]["met"].append(m)
            for t in range(1, len(ids)):
                lo = max(0, t - L)
                pad = L - (t - lo)
                d = ex[s]
                d["ctx"].append([-1] * pad + ids[lo:t])
                d["cmet"].append([[0, 0, 0, 0]] * pad + met[lo:t])
                d["qmet"].append(met[t])
                d["y"].append(ids[t])
                d["work"].append(p["work"])
        self.ex: dict[str, dict[str, Any]] = {}
        for s, d in ex.items():
            self.ex[s] = {k: (np.array(v) if k != "work" else v) for k, v in d.items()}
            self.ex[s]["unseen"] = ~self.in_train_target[self.ex[s]["y"]]
        self.pool: dict[str, dict[str, Any]] = {}
        for s, d in pool.items():
            pmet = np.array(d["met"], dtype=np.int64)
            self.pool[s] = {"bid": np.array(d["bid"], dtype=np.int64), "work": d["work"], "met": pmet}

    # -------------------------------------------------------------- candidates
    def candidates(self, split: str, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
        """True + 5 frequency + 5 hard + 5 uniform negatives, all block instances from other works of the
        same split, metric-matched with the frozen fallback chain. Returns (cands, types, stats)."""
        rng = np.random.default_rng(seed)
        P = self.pool[split]
        pb, pm = P["bid"], P["met"]
        wi = {w: k for k, w in enumerate(sorted(set(P["work"])))}
        pw = np.array([wi[w] for w in P["work"]], dtype=np.int64)
        keys = [lambda m: (m[0], m[1], m[3]), lambda m: (m[0], m[1]), lambda m: (int(m[1] == 0),), lambda m: ()]
        buckets: list[dict[tuple, np.ndarray]] = []
        hard_b: list[dict[tuple, np.ndarray]] = []
        uniq: list[dict[tuple, np.ndarray]] = []  # distinct block ids per bucket
        for kf in keys:
            g: dict[tuple, list[int]] = defaultdict(list)
            for r in range(len(pb)):
                g[kf(pm[r])].append(r)
            buckets.append({k: np.array(v) for k, v in g.items()})
            h: dict[tuple, list[int]] = defaultdict(list)
            for k, rows in g.items():
                for r in rows:
                    h[k + (int(self.bass_pc[pb[r]]), int(self.sop_pc[pb[r]]))].append(r)
            hard_b.append({k: np.array(v) for k, v in h.items()})
            uniq.append({k: np.unique(pb[v]) for k, v in buckets[-1].items()})
        # block id -> set of works in this split (for uniform-over-ids "from another work" check)
        id_works: dict[int, set[int]] = defaultdict(set)
        for r in range(len(pb)):
            id_works[int(pb[r])].add(int(pw[r]))
        e = self.ex[split]
        y, qm = e["y"], e["qmet"]
        ew = np.array([wi[w] for w in e["work"]], dtype=np.int64)
        n = len(y)
        out = np.zeros((n, K), dtype=np.int64)
        types = np.zeros((n, K), dtype=np.int64)
        st: Counter[str] = Counter()

        def draw_inst(arr: np.ndarray, w: int, seen: set[int], tries: int = 40) -> int | None:
            if len(arr) == 0:
                return None
            for r in arr[rng.integers(0, len(arr), size=tries)]:
                b = int(pb[r])
                if pw[r] != w and b not in seen:
                    return b
            ok = arr[(pw[arr] != w) & ~np.isin(pb[arr], list(seen))]  # exhaustive fallback
            return int(pb[ok[rng.integers(len(ok))]]) if len(ok) else None

        def draw_id(arr: np.ndarray, w: int, seen: set[int], tries: int = 40) -> int | None:
            if len(arr) == 0:
                return None
            for b in arr[rng.integers(0, len(arr), size=tries)]:
                b = int(b)
                if b not in seen and (id_works[b] - {w}):
                    return b
            ok = [int(b) for b in arr if int(b) not in seen and (id_works[int(b)] - {w})]  # exhaustive fallback
            return ok[int(rng.integers(len(ok)))] if ok else None

        for r in range(n):
            t, w, m = int(y[r]), int(ew[r]), qm[r]
            chosen, typ, seen = [t], [0], {t}
            hk = (int(self.bass_pc[t]), int(self.sop_pc[t]))
            for gt, cnt in ((1, 5), (2, 5), (3, 5)):
                for _ in range(cnt):
                    b: int | None = None
                    tt = gt
                    for lv, kf in enumerate(keys):
                        kk = kf(m)
                        if gt == 1:
                            b = draw_inst(buckets[lv].get(kk, np.zeros(0, np.int64)), w, seen)
                        elif gt == 2:
                            b = draw_inst(hard_b[lv].get(kk + hk, np.zeros(0, np.int64)), w, seen)
                        else:
                            b = draw_id(uniq[lv].get(kk, np.zeros(0, np.int64)), w, seen)
                        if b is not None:
                            st[f"level{lv}"] += 1
                            break
                    if b is None and gt == 2:  # hard pool exhausted: top up with a uniform draw
                        tt = 3
                        st["hard_topup"] += 1
                        for lv, kf in enumerate(keys):
                            b = draw_id(uniq[lv].get(kf(m), np.zeros(0, np.int64)), w, seen)
                            if b is not None:
                                st[f"level{lv}"] += 1
                                break
                    if b is None:
                        raise RuntimeError(f"cannot fill candidate set ({split}, row {r})")
                    chosen.append(b); typ.append(tt); seen.add(b)
            perm = rng.permutation(K)
            out[r] = np.array(chosen)[perm]
            types[r] = np.array(typ)[perm]
        tot = n * (K - 1)
        stats = {"negatives": tot, "metric_fallback_rate": round(1 - st["level0"] / tot, 5),
                 "levels": {k: st[k] for k in sorted(st) if k.startswith("level")},
                 "hard_topup_rate": round(st["hard_topup"] / (n * 5), 5)}
        return out, types, stats


# ----------------------------------------------------------------------------- baselines
class SuffixR1:
    """Factor-oracle-style longest-suffix continuation over block ids (frozen R1 score)."""

    def __init__(self, data: Data) -> None:
        self.tab: dict[tuple, Counter] = defaultdict(Counter)
        self.uni: Counter[int] = Counter()
        for s in data.seqs["train"]:
            for t in range(len(s)):
                self.uni[s[t]] += 1
                for k in range(1, L + 1):
                    if t - k < 0:
                        break
                    self.tab[tuple(s[t - k:t])][s[t]] += 1

    def score(self, data: Data, split: str, cands: np.ndarray) -> np.ndarray:
        e = data.ex[split]
        out = np.zeros(cands.shape)
        for r in range(len(cands)):
            ctx = [int(c) for c in e["ctx"][r] if c >= 0]
            for j, c in enumerate(cands[r]):
                c = int(c)
                u = self.uni.get(c, 0)
                if u == 0:
                    continue  # never seen in train: 0
                kbest, cnt = 0, 0
                for k in range(1, min(L, len(ctx)) + 1):
                    n = self.tab.get(tuple(ctx[-k:]))
                    if n is None or n.get(c, 0) == 0:
                        break  # a longer suffix + c cannot occur if this one does not
                    kbest, cnt = k, n[c]
                out[r, j] = kbest + 0.01 * math.log1p(cnt) + 1e-4 * math.log1p(u)
        return out


def rules_r2(data: Data, split: str, cands: np.ndarray) -> np.ndarray:
    """Cope-style matcher: voice-leading distance, parallel 5ths/8ves, crossings (frozen weights).
    Implementation detail: parallels need both voices to move in the same direction with the same
    interval class (P1/P8 or P5) at both slots; crossings count adjacent-voice inversions."""
    e = data.ex[split]
    prev = e["ctx"][:, -1]
    out = np.zeros(cands.shape)
    V = data.vals
    for r in range(len(cands)):
        p = int(prev[r])
        if p < 0:
            continue  # piece start: no previous block, all candidates tie (broken by jitter)
        last = []
        lastslot = []
        for v in range(4):
            seq = [int(x) for x in V[p, v * 4:v * 4 + 4] if x >= 24]
            last.append(seq[-1] if seq else -1)
            nonpad = [int(x) for x in V[p, v * 4:v * 4 + 4] if x != 0]
            lastslot.append(nonpad[-1] if nonpad else 1)
        for j, c in enumerate(cands[r]):
            first = [int(V[c, v * 4]) for v in range(4)]
            dist = sum(12 if (f < 24 or l < 24) else abs(f - l) for f, l in zip(first, last))
            par = 0
            for a in range(4):
                for b in range(a + 1, 4):
                    pa, pb_, ca, cb = lastslot[a], lastslot[b], first[a], first[b]
                    if min(pa, pb_, ca, cb) < 24:
                        continue
                    ia, ib = (pa - pb_) % 12, (ca - cb) % 12
                    if ia == ib and ia in (0, 7) and ca != pa and cb != pb_ and (ca - pa) * (cb - pb_) > 0:
                        par += 1
            cross = sum(1 for a in range(3) if first[a] >= 24 and first[a + 1] >= 24 and first[a] < first[a + 1])
            out[r, j] = -dist - 6 * par - 3 * cross
    return out


# ----------------------------------------------------------------------------- neural arms
class Model(nn.Module):
    def __init__(self, data: Data, arm: str, d: int = 256, layers: int = 4, heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.arm = arm
        self.d = d
        self.tok: torch.Tensor
        self.vcode: torch.Tensor
        self.ons: torch.Tensor
        self.pcs: torch.Tensor
        self.register_buffer("tok", torch.tensor(data.tok))
        self.register_buffer("vcode", torch.tensor(data.vcode))
        self.register_buffer("ons", torch.tensor(data.ons))
        self.register_buffer("pcs", torch.tensor(data.pcs))
        self.e_tok = nn.Embedding(data.V, d)
        self.e_val = nn.Embedding(NV, 32)
        self.e_on = nn.Embedding(2, 32)
        self.cell_proj = nn.Linear(16 * 64, d)
        self.e_pcs = nn.Linear(24, d)
        self.e_bpb = nn.Embedding(7, d)
        self.e_bi = nn.Embedding(6, d)
        self.e_comp = nn.Embedding(2, d)
        self.e_mode = nn.Embedding(2, d)
        self.e_pos = nn.Embedding(L + 2, d)
        self.e_type = nn.Embedding(3, d)  # 0 ctx, 1 query, 2 candidate
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        if arm == "B3":
            self.cand_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            self.proj = nn.Linear(d, d)
        else:
            self.head = nn.Linear(d, 1)

    def block(self, idx: torch.Tensor) -> torch.Tensor:
        i = idx.clamp(min=0)
        t = self.tok[i]
        if self.training:
            t = torch.where(torch.rand(t.shape, device=t.device) < TOK_DROP, torch.ones_like(t), t)
        cells = torch.cat([self.e_val(self.vcode[i]), self.e_on(self.ons[i])], -1)  # [...,16,64]
        return self.e_tok(t) + self.cell_proj(cells.flatten(-2)) + self.e_pcs(self.pcs[i])

    def metric(self, m: torch.Tensor) -> torch.Tensor:
        return self.e_bpb(m[..., 0]) + self.e_bi(m[..., 1]) + self.e_comp(m[..., 2]) + self.e_mode(m[..., 3])

    def context(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = b["ctx"]
        pos = torch.arange(L, device=ctx.device)
        h = self.block(ctx) + self.metric(b["cmet"]) + self.e_pos(pos) + self.e_type.weight[0]
        return h, ctx < 0

    def query(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.metric(b["qmet"]) + self.e_pos.weight[L] + self.e_type.weight[1]

    def forward(self, b: dict[str, torch.Tensor], cands: torch.Tensor) -> torch.Tensor:
        B, Kc = cands.shape
        if self.arm == "B3":
            h, m = self.context(b)
            x = torch.cat([h, self.query(b)[:, None]], 1)
            m = torch.cat([m, torch.zeros(B, 1, dtype=torch.bool, device=m.device)], 1)
            z = self.norm(self.enc(x, src_key_padding_mask=m))[:, -1]
            c = self.cand_mlp(self.block(cands))
            return torch.einsum("bd,bkd->bk", self.proj(z), c) / math.sqrt(self.d)
        cand = self.block(cands) + self.query(b)[:, None] + self.e_pos.weight[L + 1] + self.e_type.weight[2]
        if self.arm == "C1":
            z = self.norm(self.enc(cand.reshape(B * Kc, 1, self.d)))[:, -1]
        else:
            h, m = self.context(b)
            x = torch.cat([h[:, None].expand(B, Kc, L, self.d), cand[:, :, None]], 2).reshape(B * Kc, L + 1, self.d)
            m = torch.cat([m, torch.zeros(B, 1, dtype=torch.bool, device=m.device)], 1)
            m = m[:, None].expand(B, Kc, L + 1).reshape(B * Kc, L + 1)
            z = self.norm(self.enc(x, src_key_padding_mask=m))[:, -1]
        return self.head(z).view(B, Kc)


def to_batch(e: dict, rows: np.ndarray, dev: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(e[k][rows], device=dev) for k in ("ctx", "cmet", "qmet", "y")}


@torch.no_grad()
def score_neural(model: Model, data: Data, split: str, cands: np.ndarray, dev: torch.device,
                 bs: int = 512) -> np.ndarray:
    model.eval()
    e = data.ex[split]
    out = np.zeros(cands.shape, dtype=np.float32)
    for i in range(0, len(cands), bs):
        r = np.arange(i, min(i + bs, len(cands)))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            s = model(to_batch(e, r, dev), torch.as_tensor(cands[r], device=dev))
        out[r] = s.float().cpu().numpy()
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
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * math.ceil(n / bs),
                                                pct_start=0.1)
    rng = np.random.default_rng(seed)
    best: tuple[float, dict[str, torch.Tensor], int] = (float("inf"), {}, -1)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        ce = train_cands[ep % len(train_cands)]
        ypos_all = (ce == e["y"][:, None]).argmax(1)
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            r = perm[i:i + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                s = model(to_batch(e, r, dev), torch.as_tensor(ce[r], device=dev))
                loss = F.cross_entropy(s.float(), torch.as_tensor(ypos_all[r], device=dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(r)
        m = kset_metrics(score_neural(model, data, "dev", dev_cands, dev), dev_cands, data.ex["dev"]["y"])
        hist.append({"epoch": ep, "train_loss": tot / n, "dev_nll": m["nll"], "dev_top1": m["top1"]})
        if m["nll"] < best[0]:
            best = (m["nll"], {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
        if time.time() - t0 > 20 * 60:
            log(f"  {arm} seed{seed} lr{lr}: 20-min budget hit at epoch {ep}")
            break
    model.load_state_dict(best[1])
    return model, {"history": hist, "best_epoch": best[2], "train_seconds": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------- metrics
def metrics(scores: np.ndarray, cands: np.ndarray, types: np.ndarray, y: np.ndarray,
            mask: np.ndarray | None = None) -> dict:
    m = kset_metrics(scores, cands, y, mask)
    if m["n"] == 0:
        return m
    if mask is not None:
        scores, cands, types, y = scores[mask], cands[mask], types[mask], y[mask]
    s = scores + jitter(scores)
    ypos = (cands == y[:, None]).argmax(1)
    ly = s[np.arange(len(y)), ypos]
    rank = (s > ly[:, None]).sum(1)
    m["mrr"] = float((1.0 / (rank + 1)).mean())
    p = np.exp(logp(s))
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), ypos] = 1
    m["brier"] = float(((p - onehot) ** 2).sum(1).mean())
    top = types[np.arange(len(y)), s.argmax(1)]
    for k, nm in enumerate(NEG_TYPES[1:], 1):
        m[f"err_{nm}"] = float((top == k).mean())  # share of decisions won by a negative of this type
    return m


def env_info(dev: torch.device, data_path: Path) -> dict:
    def git(*a: str) -> str:
        try:
            return subprocess.check_output(["git", *a], cwd=HERE, text=True).strip()
        except Exception:  # noqa: BLE001
            return "?"
    man = HERE / "split-manifest-B.json"
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
            "platform": platform.platform(), "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", ".")),
            "pieces_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(man.read_bytes()).hexdigest() if man.exists() else None,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global RES, MODELS
    data_path = HERE / "data" / "stageB" / "pieces.json"
    if args.smoke:
        RES, MODELS = HERE / "results-b-smoke", HERE / "models-b-smoke"
        data_path = HERE / "data" / "stageB" / "smoke" / "pieces.json"
    frozen_path = RES / "FROZEN-B.json" if args.smoke else HERE / "FROZEN-B.json"
    RES.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    (RES / "run.log").write_text("", encoding="utf-8")
    t_start = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    epochs = 2 if args.smoke else 15
    seeds = [0] if args.smoke else SEEDS
    lrs = [1e-3] if args.smoke else LRS
    env = env_info(dev, data_path)
    log(f"env {json.dumps(env)}  smoke={args.smoke}")
    status("loading data")
    data = Data(data_path)
    sizes = {s: {"decisions": len(data.ex[s]["y"]), "works": len(set(data.ex[s]["work"])),
                 "unseen": int(data.ex[s]["unseen"].sum())} for s in data.ex}
    log(f"blocks {data.N}, vocab {data.V}, sizes {json.dumps(sizes)}")
    status("sampling candidate sets")
    train_cs = [data.candidates("train", SEED_DATA + 100 + i) for i in range(4)]
    train_cands = [c for c, _, _ in train_cs]
    cs = {s: data.candidates(s, SEED_DATA + j) for j, s in enumerate(SPLITS_EVAL)}
    cands = {s: cs[s][0] for s in SPLITS_EVAL}
    types = {s: cs[s][1] for s in SPLITS_EVAL}
    cand_stats = {"train": [st for _, _, st in train_cs], **{s: cs[s][2] for s in SPLITS_EVAL}}
    log(f"candidate stats {json.dumps(cand_stats)}")

    scores: dict[str, dict[str, list[np.ndarray]]] = {s: defaultdict(list) for s in SPLITS_EVAL}
    r1 = SuffixR1(data)

    def score_baselines(split: str) -> None:
        scores[split]["R0"].append(np.random.default_rng(1).random(cands[split].shape))
        scores[split]["R1"].append(r1.score(data, split, cands[split]))
        scores[split]["R2"].append(rules_r2(data, split, cands[split]))

    status("baselines on dev")
    score_baselines("dev")
    ydev = data.ex["dev"]["y"]
    dev_base = {a: metrics(scores["dev"][a][0], cands["dev"], types["dev"], ydev) for a in ("R0", "R1", "R2")}
    log("dev baselines " + json.dumps({a: round(m["top1"], 4) for a, m in dev_base.items()}))

    # ---- dev-only lr selection on seed 0
    dev_sel: dict[str, dict] = {}
    seed0: dict[str, Model] = {}
    for arm in NEURAL:
        res = {}
        models: dict[str, Model] = {}
        for lr in lrs:
            status(f"dev selection {arm} lr={lr}")
            model, info = train_neural(data, arm, 0, lr, dev, epochs, train_cands, cands["dev"])
            m = metrics(score_neural(model, data, "dev", cands["dev"], dev), cands["dev"], types["dev"], ydev)
            res[str(lr)] = {"dev": m, **info}
            models[str(lr)] = model
            log(f"  {arm} lr={lr} dev top1={m['top1']:.4f} nll={m['nll']:.4f} best_ep={info['best_epoch']}")
        best_lr = min(res, key=lambda k: res[k]["dev"]["nll"])
        dev_sel[arm] = {"grid": res, "lr": float(best_lr)}
        seed0[arm] = models[best_lr]  # seed-0 run at the selected lr is reused as the final seed 0
        del models
        torch.cuda.empty_cache()
    best_dev = {a: dev_sel[a]["grid"][str(dev_sel[a]["lr"])]["dev"]["top1"] for a in NEURAL}
    r1d, r2d = dev_base["R1"]["top1"], dev_base["R2"]["top1"]
    gate_ok = bool(best_dev["C1"] <= r1d + 0.05 and best_dev["C1"] <= r2d + 0.05)
    beats_both = {a: bool(best_dev[a] > r1d and best_dev[a] > r2d) for a in ("V1", "B3")}
    stop = not (beats_both["V1"] or beats_both["B3"])
    (RES / "stageB-dev.json").write_text(json.dumps({"baselines": dev_base, "neural": dev_sel, "sizes": sizes,
                                                     "candidate_stats": cand_stats}, indent=1), encoding="utf-8")
    frozen = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "env": env, "epochs": epochs, "K": K, "L": L,
              "seeds": seeds, "lr": {a: dev_sel[a]["lr"] for a in NEURAL},
              "dev_top1": {"R0": dev_base["R0"]["top1"], "R1": r1d, "R2": r2d, **best_dev},
              "validity_gate": {"rule": "C1 <= R1 + 0.05 and C1 <= R2 + 0.05 (dev top-1, seed 0)", "pass": gate_ok},
              "stop_rule": {"rule": "stop if neither V1 nor B3 beats both R1 and R2 on dev",
                            "beats_both": beats_both, "triggered": stop},
              "selection": "lr by dev K=16 NLL on seed 0; epoch by dev NLL",
              "hypotheses": {"HB1": "V1 - B3 test_ood top-1 CI lower > 0",
                             "HB2": "V1 - max(R1,R2) test_ood top-1 CI lower > 0 (max = better on test_ood)",
                             "HB3": "V1 - R1 on test_ood unseen slice CI lower > 0"}}
    frozen_path.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    log(f"FROZEN-B written: lr={frozen['lr']} gate_pass={gate_ok} stop={stop} dev_top1={json.dumps(frozen['dev_top1'])}")
    if (not gate_ok or stop) and not args.smoke:  # smoke exercises the test path regardless
        why = "validity gate failed (C1 leak)" if not gate_ok else "stop rule (no learned arm beats R1 and R2)"
        status(f"STOPPED before test: {why}")
        write_report(None, frozen, dev_base, dev_sel, sizes, cand_stats, env, t_start)
        return

    # ---- final: seeds x selected lr, score once
    status("baselines on test splits")
    for s in ("test_id", "test_ood"):
        score_baselines(s)
    train_info: dict[str, list] = defaultdict(list)
    for arm in NEURAL:
        for seed in seeds:
            status(f"final {arm} seed={seed}")
            if seed == 0:
                model = seed0.pop(arm)
                info = dev_sel[arm]["grid"][str(dev_sel[arm]["lr"])]
            else:
                model, info = train_neural(data, arm, seed, dev_sel[arm]["lr"], dev, epochs, train_cands,
                                           cands["dev"])
            train_info[arm].append({"seed": seed, "best_epoch": info["best_epoch"],
                                    "train_seconds": info["train_seconds"]})
            torch.save(model.state_dict(), MODELS / f"{arm}-seed{seed}.pt")
            for s in SPLITS_EVAL:
                scores[s][arm].append(score_neural(model, data, s, cands[s], dev))
            del model
            torch.cuda.empty_cache()

    status("metrics + bootstrap")
    arms = ["R0", "R1", "R2"] + NEURAL
    results: dict[str, Any] = {"kset": {}, "bootstrap": {}, "sizes": sizes, "candidate_stats": cand_stats,
                               "train_info": train_info, "frozen": frozen, "env": env}
    corr: dict[str, dict[str, np.ndarray]] = {}
    for s in SPLITS_EVAL:
        e = data.ex[s]
        y, uns = e["y"], e["unseen"]
        results["kset"][s] = {}
        for a in arms:
            results["kset"][s][a] = {
                "all": summarise([metrics(sc, cands[s], types[s], y) for sc in scores[s][a]]),
                "seen": summarise([metrics(sc, cands[s], types[s], y, ~uns) for sc in scores[s][a]]),
                "unseen": summarise([metrics(sc, cands[s], types[s], y, uns) for sc in scores[s][a]])}
        corr[s] = {a: np.mean([per_example_correct(sc, cands[s], y) for sc in scores[s][a]], 0) for a in arms}
        w = e["work"]
        c = corr[s]
        best_r = "R1" if c["R1"].mean() >= c["R2"].mean() else "R2"
        bt: dict[str, Any] = {"V1_minus_B3": cluster_bootstrap(c["V1"] - c["B3"], w, seed=SEED_DATA),
                              f"V1_minus_maxR({best_r})": cluster_bootstrap(c["V1"] - c[best_r], w, seed=SEED_DATA),
                              "V1_minus_R1": cluster_bootstrap(c["V1"] - c["R1"], w, seed=SEED_DATA),
                              "V1_minus_R2": cluster_bootstrap(c["V1"] - c["R2"], w, seed=SEED_DATA),
                              "B3_minus_maxR": cluster_bootstrap(c["B3"] - c[best_r], w, seed=SEED_DATA),
                              "V1_minus_C1": cluster_bootstrap(c["V1"] - c["C1"], w, seed=SEED_DATA),
                              "best_R": best_r}
        for nm, msk in (("seen", ~uns), ("unseen", uns)):
            if msk.sum() > 0:
                ww = [w[i] for i in np.where(msk)[0]]
                bt[f"V1_minus_R1_{nm}"] = cluster_bootstrap((c["V1"] - c["R1"])[msk], ww, seed=SEED_DATA)
                bt[f"V1_minus_B3_{nm}"] = cluster_bootstrap((c["V1"] - c["B3"])[msk], ww, seed=SEED_DATA)
        results["bootstrap"][s] = bt
    o = results["bootstrap"]["test_ood"]
    hb2_key = f"V1_minus_maxR({o['best_R']})"
    results["H"] = {
        "HB1": {"ci": o["V1_minus_B3"], "supported": bool(o["V1_minus_B3"]["ci95"][0] > 0)},
        "HB2": {"ci": o[hb2_key], "max_R": o["best_R"], "supported": bool(o[hb2_key]["ci95"][0] > 0)},
        "HB3": {"ci": o.get("V1_minus_R1_unseen"),
                "supported": bool(o.get("V1_minus_R1_unseen", {"ci95": [0]})["ci95"][0] > 0)}}
    results["wall_seconds"] = round(time.time() - t_start, 1)
    results["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else None
    (RES / "stageB-results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    write_report(results, frozen, dev_base, dev_sel, sizes, cand_stats, env, t_start)
    h = results["H"]
    status(f"DONE in {results['wall_seconds'] / 60:.1f} min; HB1={h['HB1']['supported']} "
           f"HB2={h['HB2']['supported']} HB3={h['HB3']['supported']}")


def write_report(results: dict | None, frozen: dict, dev_base: dict, dev_sel: dict, sizes: dict,
                 cand_stats: dict, env: dict, t0: float) -> None:
    L_ = ["# Stage B step 1 report: candidate-conditioned next beat block (preliminary pilot)", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by run_stage_b.py. Commit {env['commit']} "
          f"(dirty={env['dirty']}). Device {env['device']}, torch {env['torch']}. "
          f"Wall {(time.time() - t0) / 60:.1f} min.", "",
          "Task: pick the true next beat block (4 voices x 4 slots) among K=16 metric-matched block instances "
          "from other works (5 frequency / 5 hard / 5 uniform). Train/dev/test_id: music21 Bach chorales "
          "(work-clustered split). test_ood: 150 OpenScore Lieder piano accompaniments reduced to SATB.", "",
          "Sizes: " + ", ".join(f"{s} {v['decisions']} decisions / {v['works']} works / {v['unseen']} unseen"
                                for s, v in sizes.items()), "",
          "Metric fallback rate: " + ", ".join(f"{s} {cand_stats[s]['metric_fallback_rate']}"
                                               for s in SPLITS_EVAL) + "; hard top-up rate: " +
          ", ".join(f"{s} {cand_stats[s]['hard_topup_rate']}" for s in SPLITS_EVAL), "",
          "## Dev selection (seed 0)", "", "| arm | lr | dev top-1 | dev NLL |", "|---|---|---|---|"]
    for a, m in dev_base.items():
        L_.append(f"| {a} | - | {m['top1']:.3f} | {m['nll']:.3f} |")
    for a, v in dev_sel.items():
        for lr, g in v["grid"].items():
            sel = " (selected)" if float(lr) == v["lr"] else ""
            L_.append(f"| {a} | {lr}{sel} | {g['dev']['top1']:.3f} | {g['dev']['nll']:.3f} |")
    L_ += ["", f"Validity gate (C1 <= R1+5pp and <= R2+5pp): pass={frozen['validity_gate']['pass']}. "
           f"Stop rule triggered: {frozen['stop_rule']['triggered']} (beats both R1 and R2: "
           f"{frozen['stop_rule']['beats_both']}).", ""]
    if results is not None:
        for s in SPLITS_EVAL:
            L_ += [f"## {s} (mean ± sd over seeds)", "",
                   "| arm | top-1 | top-3 | MRR | NLL | ECE | top-1 seen | top-1 unseen | won by freq/hard/unif |",
                   "|---|---|---|---|---|---|---|---|---|"]
            for a, v in results["kset"][s].items():
                al, se, un = v["all"], v["seen"], v["unseen"]
                fs = lambda d: f"{d['top1']:.3f} (n={d['n']})" if d.get("n") else "-"  # noqa: E731
                L_.append(f"| {a} | {al['top1']:.3f}±{al['top1_sd']:.3f} | {al['top3']:.3f} | {al['mrr']:.3f} | "
                          f"{al['nll']:.3f} | {al['ece15']:.3f} | {fs(se)} | {fs(un)} | "
                          f"{al['err_freq']:.3f}/{al['err_hard']:.3f}/{al['err_unif']:.3f} |")
            L_ += ["", "Work-cluster bootstrap (2000 draws, seed-averaged correctness):", ""]
            for k, b in results["bootstrap"][s].items():
                if isinstance(b, dict):
                    L_.append(f"- {k}: {b['mean']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] over {b['works']} works")
            L_.append("")
        L_ += ["## Hypotheses (test_ood)", ""]
        for k, v in results["H"].items():
            ci = v["ci"]
            L_.append(f"- {k}: {frozen['hypotheses'][k]} -> {ci['mean']:+.4f} [{ci['ci95'][0]:+.4f}, "
                      f"{ci['ci95'][1]:+.4f}]; supported={v['supported']}" if ci else f"- {k}: n/a")
        L_ += ["", f"Peak GPU memory {results['peak_gpu_mem_gb']} GB.", ""]
    L_ += ["Caveats: preliminary pilot; block continuation is corpus plausibility, not musical quality; "
           "no listening evidence; Lieder SATB reduction is part of the corpus shift."]
    (RES / "stageB-report.md").write_text("\n".join(L_) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:  # noqa: BLE001
        import traceback
        status(f"FAILED: {type(ex).__name__}: {ex}")
        log(traceback.format_exc())
        raise
