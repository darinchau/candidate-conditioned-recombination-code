"""Stage E3e: chained recombination scored with root-free pc-set harmony metrics.

Frozen spec: plan-stageE3E-frozen.md. Usage:
    .venv-gpu/Scripts/python.exe run_stage_e3e.py --audit     # CPU audits, writes results-e3e-audit/
    .venv-gpu/Scripts/python.exe run_stage_e3e.py --smoke     # 6 prompts per split, seed 0, results-e3e-smoke/
    .venv-gpu/Scripts/python.exe run_stage_e3e.py             # full run, results-e3e/"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_stage_bl as bl
import run_stage_bn3 as bn
from run_stage_a import cluster_bootstrap
from run_stage_b import L, SEED_DATA, SEEDS, Data, Model
from run_stage_b2 import EVAL

HERE = Path(__file__).resolve().parent
STEPS = bl.STEPS
K = 16
LAMBDAS = [0.5, 1.0, 2.0, 4.0]
VL_TOL = 1.10
BOOT_SEED = 20261012
SEL_SPLITS = ("dev", "dev_lied")
TEST_SPLITS = ("test_id", "test_lied")
MODEL_DIR = HERE / "models-r2" / "bn3" / "S2"
BASE_ARMS = ["R0", "R2", "B3", "V1", "O-pc"]
MAJOR = {0, 2, 4, 5, 7, 9, 11}
MINOR = {0, 2, 3, 5, 7, 8, 10, 11}  # natural + raised leading tone
METRICS = ["trig_lp", "jaccard", "diatonic", "hr_gap", "vl_dist", "parallels", "crossings", "continuity"]
CONTRASTS = [("V1", "B3"), ("H-V", "V1"), ("H-R", "R2"), ("O-pc", "V1")]
BOS = -1


def pcset(vals: np.ndarray) -> int:
    m = 0
    for v in vals:
        if v >= 2:
            m |= 1 << (int(v) % 12)
    return m


def pcs(mask: int) -> set[int]:
    return {i for i in range(12) if mask >> i & 1}


def jaccard(a: int, b: int) -> float:
    u = bin(a | b).count("1")
    return 1.0 if u == 0 else bin(a & b).count("1") / u


class WB:
    """Interpolated Witten-Bell n-gram over pc-set masks, base = uniform over 4096 masks."""

    def __init__(self, seqs: list[list[int]], n: int) -> None:
        self.n = n
        self.c: list[dict[tuple[int, ...], Counter[int]]] = [defaultdict(Counter) for _ in range(n)]
        for s in seqs:
            x = [BOS] * (n - 1) + s
            for i in range(n - 1, len(x)):
                for o in range(n):
                    self.c[o][tuple(x[i - o:i])][x[i]] += 1
        self.tot = [{h: sum(v.values()) for h, v in co.items()} for co in self.c]

    def prob(self, hist: list[int], y: int) -> float:
        p = 1.0 / 4096
        for o in range(self.n):
            h = tuple(hist[len(hist) - o:]) if o else ()
            if len(h) < o:
                continue
            ct = self.c[o].get(h)
            if not ct:
                continue
            n, t = self.tot[o][h], len(ct)
            p = (ct.get(y, 0) + t * p) / (n + t)
        return p


def diatonic(mask: int, mode: int) -> float:
    s = pcs(mask)
    return float("nan") if not s else len(s & (MINOR if mode else MAJOR)) / len(s)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@torch.no_grad()
def logits(model: torch.nn.Module, hist: list[int], hmet: list[list[int]], qmet: list[int],
           cands: list[int], data: Data, dev: torch.device) -> np.ndarray:
    assert len(hist) == L and len(hmet) == L
    b = {"ctx": torch.tensor([hist], device=dev), "cmet": torch.tensor([hmet], device=dev),
         "qmet": torch.tensor([qmet], device=dev), "y": torch.zeros(1, dtype=torch.long, device=dev)}
    s = model(b, torch.tensor([cands], device=dev))[0]
    return torch.log_softmax(s.float(), 0).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    bn.TABLE = "r2"
    res = HERE / ("results-e3e-audit" if a.audit else "results-e3e-smoke" if a.smoke else "results-e3e")
    res.mkdir(exist_ok=True)
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.audit else "cpu")
    dpath = bn.cell_path("S2", False)
    data = Data(dpath, EVAL)
    S = np.array([pcset(v) for v in data.vals], dtype=np.int64)
    assert np.array_equal(S, np.array([pcset(v) for v in data.vals])), "nondeterministic pc-set"
    bidx = {b: i for i, b in enumerate(data.bids)}
    assert all((p["tonic_pc"] + p["shift"]) % 12 == 0 for p in data.pieces), "not key-normalised"

    train = [[int(S[bidx[b]]) for b in p["bids"]] for p in data.pieces if p["split"] == "train"]
    tri, big = WB(train, 3), WB(train, 2)

    seen_w: set[str] = set()
    prompts: dict[str, list[dict[str, Any]]] = {s: [] for s in SEL_SPLITS + TEST_SPLITS}
    for p in data.pieces:
        if p["split"] in prompts and p["work"] not in seen_w and len(p["bids"]) >= L + STEPS:
            seen_w.add(p["work"])
            prompts[p["split"]].append(p)
    counts = {s: len(v) for s, v in prompts.items()}
    expect = {"dev": 26, "dev_lied": 46, "test_id": 20, "test_lied": 116}
    assert counts == expect, f"prompt counts {counts} != {expect}"
    if a.smoke:
        prompts = {s: v[:6] for s, v in prompts.items()}
    # Test prompts keep chain_hd.py's index (data.pieces order over test_id+test_lied) so pools match
    # results-r2 HD chaining; selection prompts are offset by 10000.
    tlist = [p for p in data.pieces if any(p is q for s in TEST_SPLITS for q in prompts[s])]
    slist = [p for p in data.pieces if any(p is q for s in SEL_SPLITS for q in prompts[s])]
    pidx = {p["id"]: i for i, p in enumerate(tlist)} | {p["id"]: 10000 + i for i, p in enumerate(slist)}

    ckpts = {f"{arm}-seed{sd}": MODEL_DIR / f"{arm}-seed{sd}.pt" for arm in ("V1", "B3") for sd in SEEDS}
    manifest: dict[str, Any] = {
        "written": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "audit" if a.audit else "smoke" if a.smoke else "full",
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
        "dirty": bool(subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True,
                                     cwd=HERE).stdout.strip()),
        "sha256": {"data": sha256(dpath), "runner": sha256(Path(__file__)),
                   **{k: sha256(v) for k, v in ckpts.items()},
                   **({"plan": sha256(HERE / "plan-stageE3E-frozen.md")}
                      if (HERE / "plan-stageE3E-frozen.md").exists() else {})},
        "prompt_counts": counts, "train_pieces": len(train), "pcset_vocab_train": len({x for s in train for x in s}),
        "K": K, "L": L, "STEPS": STEPS, "seeds": [0] if a.smoke else SEEDS, "lambdas": LAMBDAS,
    }

    pools = {s: bl.Pool(data, s) for s in prompts}
    bl.K = K
    draws: dict[str, list[list[int]]] = {}
    for s in prompts:
        for p in prompts[s]:
            met = [[m[0], m[1], m[2], p["mode"]] for m in p["beats"]]
            pi = pidx[p["id"]]
            rows = [pools[s].draw(met[t], p["work"], np.random.default_rng([SEED_DATA, pi, t]))
                    for t in range(L, L + STEPS)]
            again = [pools[s].draw(met[t], p["work"], np.random.default_rng([SEED_DATA, pi, t]))
                     for t in range(L, L + STEPS)]
            assert all(list(x) == list(y) for x, y in zip(rows, again)), "pool draw nondeterministic"
            for r in rows:
                assert len(r) == K and all(pools[s].work[j] != p["work"] for j in r), "pool leak / size"
            draws[p["id"]] = [[int(j) for j in r] for r in rows]
    manifest["pool_sha256"] = hashlib.sha256(json.dumps(draws, sort_keys=True).encode()).hexdigest()

    seeds = [0] if a.smoke else SEEDS
    models: dict[str, list[torch.nn.Module]] = {}
    for arm in ("V1", "B3"):
        ms: list[torch.nn.Module] = []
        for sd in seeds:
            m = Model(data, arm).to(dev)
            m.load_state_dict(torch.load(ckpts[f"{arm}-seed{sd}"], map_location=dev))
            ms.append(m.eval())
        models[arm] = ms

    if a.audit:
        manifest["prompt_blocks_total"] = sum(len(p["bids"]) for s in prompts for p in prompts[s])
        manifest["pool_candidates_in_vocab"] = float(np.mean(
            [bool(data.in_train_target[int(pools[s].pb[j])]) for s in prompts for p in prompts[s]
             for r in draws[p["id"]] for j in r]))
        s_dev = [[int(S[bidx[b]]) for b in p["bids"]] for p in data.pieces if p["split"] == "dev"]
        lps = [math.log2(tri.prob(x[max(0, i - 2):i] if i >= 2 else [BOS] * (2 - i) + x[:i], x[i]))
              for x in s_dev for i in range(len(x))]
        manifest["dev_trigram_mean_log2p"] = float(np.mean(lps))
        manifest["dev_unigram_oov_rate"] = float(np.mean([y not in tri.c[0][()] for x in s_dev for y in x]))
        (res / "audit.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print(json.dumps(manifest, indent=1))
        return

    def run_prompt(p: dict[str, Any], arm: str, lam: float, model: torch.nn.Module | None) -> dict[str, float]:
        s = p["split"]
        ids = [bidx[b] for b in p["bids"]]
        met = [[m[0], m[1], m[2], p["mode"]] for m in p["beats"]]
        hist = ids[:L]
        chosen: list[int] = []
        rng = np.random.default_rng([SEED_DATA, pidx[p["id"]], 999])
        orig = ids[L:L + STEPS]
        for k, t in enumerate(range(L, L + STEPS)):
            rows = draws[p["id"]][k]
            cands = [int(pools[s].pb[j]) for j in rows]
            if arm == "R0":
                j = int(rng.integers(K))
            elif arm == "O-pc":
                jac = [jaccard(int(S[c]), int(S[orig[k]])) for c in cands]
                j = int(np.argmax(jac))
            else:
                if arm in ("R2", "H-R"):
                    sc = np.array([-d - 6 * pa - 3 * cr for d, pa, cr in (bl.r2_parts(data, hist[-1], c) for c in cands)])
                else:
                    assert model is not None
                    sc = logits(model, hist[-L:], met[t - L:t], met[t], cands, data, dev).astype(np.float64)
                if arm in ("H-R", "H-V"):
                    prev = int(S[hist[-1]])
                    sc = sc + lam * np.array([math.log(big.prob([prev], int(S[c]))) for c in cands])
                j = int(np.argmax(sc))
            hist.append(cands[j])
            chosen.append(rows[j])
        blocks = hist[L:]
        r = bl.seq_metrics(data, ids[L - 1], blocks, [pools[s].work[x] for x in chosen])
        seq = [int(S[x]) for x in ids[:L]] + [int(S[x]) for x in blocks]
        r["trig_lp"] = float(np.mean([math.log2(tri.prob(seq[i - 2:i], seq[i])) for i in range(L, L + STEPS)]))
        r["jaccard"] = float(np.mean([jaccard(int(S[x]), int(S[o])) for x, o in zip(blocks, orig)]))
        dv = [diatonic(int(S[x]), p["mode"]) for x in blocks]
        r["diatonic"] = float(np.nanmean(dv)) if not all(math.isnan(x) for x in dv) else float("nan")
        def rate(bs: list[int]) -> float:
            q = [int(S[ids[L - 1]])] + [int(S[x]) for x in bs]
            return float(np.mean([q[i] != q[i - 1] for i in range(1, len(q))]))
        r["hr_gap"] = abs(rate(blocks) - rate(orig))
        return {k: float(r[k]) for k in METRICS}

    def run_arm(ps: list[dict[str, Any]], arm: str, lam: float = 0.0) -> list[dict[str, float]]:
        base = {"H-V": "V1", "H-R": "R2"}.get(arm, arm)
        out = []
        for p in ps:
            if base in models:
                rs = [run_prompt(p, arm, lam, m) for m in models[base]]
                out.append({k: float(np.nanmean([r[k] for r in rs])) for k in METRICS})
            else:
                out.append(run_prompt(p, arm, lam, None))
        return out

    def orig_row(p: dict[str, Any]) -> dict[str, float]:
        ids = [bidx[b] for b in p["bids"]]
        orig = ids[L:L + STEPS]
        r = bl.seq_metrics(data, ids[L - 1], orig, [p["work"]] * STEPS)
        seq = [int(S[x]) for x in ids[:L + STEPS]]
        r["trig_lp"] = float(np.mean([math.log2(tri.prob(seq[i - 2:i], seq[i])) for i in range(L, L + STEPS)]))
        r["jaccard"], r["hr_gap"] = 1.0, 0.0
        r["diatonic"] = float(np.nanmean([diatonic(int(S[x]), p["mode"]) for x in orig]))
        return {k: float(r[k]) for k in METRICS}

    # ---- lambda selection on dev + dev_lied only ----
    sel = [p for s in SEL_SPLITS for p in prompts[s]]
    selrows: dict[str, Any] = {}
    for arm in ("R2", "V1"):
        selrows[arm] = run_arm(sel, arm)
    lam_out: dict[str, Any] = {"splits": list(SEL_SPLITS), "prompts": len(sel), "tolerance": VL_TOL, "grid": {}}
    chosen_lam: dict[str, float | None] = {}
    for h, base in (("H-R", "R2"), ("H-V", "V1")):
        bvl = float(np.mean([r["vl_dist"] for r in selrows[base]]))
        best: tuple[float, float] | None = None
        for lam in LAMBDAS:
            rr = run_arm(sel, h, lam)
            vl, lp = float(np.mean([r["vl_dist"] for r in rr])), float(np.mean([r["trig_lp"] for r in rr]))
            ok = vl <= VL_TOL * bvl
            lam_out["grid"][f"{h}|{lam}"] = {"vl_dist": vl, "trig_lp": lp, "base_vl_dist": bvl, "feasible": ok}
            if ok and (best is None or lp > best[1]):
                best = (lam, lp)
        chosen_lam[h] = None if best is None else best[0]
    lam_out["chosen"] = chosen_lam
    lam_out["written_before_test"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (res / "lambda-selection.json").write_text(json.dumps(lam_out, indent=1), encoding="utf-8")
    print("lambda", chosen_lam, f"{time.time() - t0:.0f}s", flush=True)

    # ---- test ----
    arms = BASE_ARMS + [h for h in ("H-R", "H-V") if chosen_lam[h] is not None]
    out: dict[str, Any] = {"manifest": manifest, "lambda": chosen_lam, "per_prompt": {}, "summary": {}, "contrasts": {}}
    for s in TEST_SPLITS:
        ps = prompts[s]
        trows = {arm: run_arm(ps, arm, chosen_lam.get(arm) or 0.0) for arm in arms}
        trows["Original"] = [orig_row(p) for p in ps]
        works = [p["work"] for p in ps]
        out["per_prompt"][s] = {"ids": [p["id"] for p in ps], "works": works, "rows": trows}
        out["summary"][s] = {arm: {k: float(np.nanmean([r[k] for r in trows[arm]])) for k in METRICS} for arm in trows}
        for x, y in CONTRASTS:
            if x not in trows or y not in trows:
                continue
            for k in METRICS:
                d = np.array([rx[k] - ry[k] for rx, ry in zip(trows[x], trows[y])])
                keep = ~np.isnan(d)
                out["contrasts"][f"{s}|{x}-{y}|{k}"] = cluster_bootstrap(
                    d[keep], [w for w, kk in zip(works, keep) if kk], draws=2000, seed=BOOT_SEED)
        print(s, f"{time.time() - t0:.0f}s", flush=True)
    out["wall_seconds"] = time.time() - t0
    (res / "chain-results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (res / "run-manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    lines = ["# Stage E3e: chained recombination, root-free pc-set harmony metrics", "",
             f"Mode {manifest['mode']}; seeds {manifest['seeds']}; K={K}; {STEPS} steps; lambda {chosen_lam}.",
             "trig_lp = mean log2 P_trigram(pc-set) (higher better); jaccard vs original continuation (higher better);",
             "diatonic share; hr_gap = |change-rate gap| to original (lower better).", ""]
    for s in TEST_SPLITS:
        lines += [f"## {s} ({len(prompts[s])} prompts)", "", "| arm | " + " | ".join(METRICS) + " |",
                  "|---" * (len(METRICS) + 1) + "|"]
        for arm, v in out["summary"][s].items():
            lines.append(f"| {arm} | " + " | ".join(f"{v[k]:.3f}" for k in METRICS) + " |")
        lines += ["", "| contrast | metric | mean | 95% CI |", "|---|---|---|---|"]
        for key, c in out["contrasts"].items():
            ss, con, k = key.split("|")
            if ss == s and k in ("trig_lp", "jaccard", "diatonic", "hr_gap", "vl_dist"):
                lines.append(f"| {con} | {k} | {c['mean']:+.3f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] |")
        lines.append("")
    (res / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
