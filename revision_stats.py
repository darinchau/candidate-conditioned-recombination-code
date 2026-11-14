"""Revision analyses for the AIMG paper.

1. TOST for the V1-B3 trig_lp difference at several equivalence margins.
2. Reference-set Jaccard: maximum overlap against real 16-step windows from
   the same piece, excluding the original continuation window.

Run with the GPU venv from the code repo root:
    ..\\..\\agent-notes\\experiments\\candidate-conditioned-decision-pilot-20261004\\.venv-gpu\\Scripts\\python.exe revision_stats.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import run_stage_bl as bl
import run_stage_bn3 as bn
import run_stage_e3e as e3e
from run_stage_a import cluster_bootstrap
from run_stage_b import L, SEED_DATA, SEEDS, Data, Model
from run_stage_b2 import EVAL

HERE = Path(__file__).resolve().parent
OUT = HERE / "results-revision"
OUT.mkdir(exist_ok=True)
K = e3e.K
STEPS = e3e.STEPS
BOOT_SEED = e3e.BOOT_SEED
MODEL_DIR = e3e.MODEL_DIR


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def tost_p(diff: np.ndarray, works: list[str], margin: float, draws: int = 2000) -> dict:
    ws = sorted(set(works))
    wi = {w: i for i, w in enumerate(ws)}
    g = np.array([wi[w] for w in works])
    sums = np.bincount(g, weights=diff, minlength=len(ws))
    cnts = np.bincount(g, minlength=len(ws)).astype(np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(ws), size=(draws, len(ws)))
    bs = sums[idx].sum(1) / cnts[idx].sum(1)
    se = float(bs.std(ddof=1))
    mean = float(diff.mean())
    # TOST: H0_lower mu <= -margin, H0_upper mu >= +margin.
    p_lower = normal_cdf((-margin - mean) / se)
    p_upper = normal_cdf((mean - margin) / se)
    return {
        "margin": margin,
        "mean": mean,
        "se": se,
        "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": max(p_lower, p_upper),
    }


def load_chain() -> tuple[np.ndarray, list[str]]:
    d = json.loads((HERE / "results-e3e" / "chain-results.json").read_text(encoding="utf-8"))
    pp = d["per_prompt"]["test_lied"]
    rows, works = pp["rows"], pp["works"]
    diff = np.array([a["trig_lp"] - b["trig_lp"] for a, b in zip(rows["V1"], rows["B3"])])
    return diff, works


def build_prompts(data: Data) -> tuple[dict, dict, dict, np.ndarray, dict]:
    S = np.array([e3e.pcset(v) for v in data.vals], dtype=np.int64)
    bidx = {b: i for i, b in enumerate(data.bids)}
    prompts: dict[str, list[dict]] = {s: [] for s in ("test_id", "test_lied")}
    seen_w: set[str] = set()
    for p in data.pieces:
        if p["split"] in prompts and p["work"] not in seen_w and len(p["bids"]) >= L + STEPS:
            seen_w.add(p["work"])
            prompts[p["split"]].append(p)
    tlist = [p for p in data.pieces if any(p is q for s in prompts for q in prompts[s])]
    pidx = {p["id"]: i for i, p in enumerate(tlist)}
    return prompts, pidx, bidx, S, {}


def run_prompt(p: dict, arm: str, model: torch.nn.Module, data: Data, S, bidx,
               pools, draws, pidx, dev: torch.device) -> dict:
    s = p["split"]
    ids = [bidx[b] for b in p["bids"]]
    met = [[m[0], m[1], m[2], p["mode"]] for m in p["beats"]]
    hist = ids[:L]
    chosen: list[int] = []
    for k, t in enumerate(range(L, L + STEPS)):
        rows = draws[p["id"]][k]
        cands = [int(pools[s].pb[j]) for j in rows]
        sc = e3e.logits(model, hist[-L:], met[t - L:t], met[t], cands, data, dev).astype(np.float64)
        j = int(np.argmax(sc))
        hist.append(cands[j])
        chosen.append(cands[j])

    # Real 16-step windows from the same piece, excluding the original window.
    ids_all = [bidx[b] for b in p["bids"]]
    refs = [ids_all[o:o + STEPS] for o in range(0, len(ids_all) - STEPS + 1)
            if o + STEPS <= L or o >= L + STEPS]
    if not refs:
        ref_j = float("nan")
    else:
        vals = []
        for t, b in enumerate(chosen):
            vals.append(max(e3e.jaccard(int(S[b]), int(S[r[t]])) for r in refs))
        ref_j = float(np.mean(vals))
    orig = ids[L:L + STEPS]
    single_j = float(np.mean([e3e.jaccard(int(S[b]), int(S[o])) for b, o in zip(chosen, orig)]))
    return {"ref_jaccard": ref_j, "jaccard": single_j}


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev)
    bn.TABLE = "r2"
    data = Data(bn.cell_path("S2", False), EVAL)
    prompts, pidx, bidx, S, _ = build_prompts(data)
    pools = {s: bl.Pool(data, s) for s in prompts}
    bl.K = K
    draws: dict[str, list[list[int]]] = {}
    for s in prompts:
        for p in prompts[s]:
            met = [[m[0], m[1], m[2], p["mode"]] for m in p["beats"]]
            pi = pidx[p["id"]]
            draws[p["id"]] = [[int(j) for j in pools[s].draw(met[t], p["work"],
                                                              np.random.default_rng([SEED_DATA, pi, t]))]
                              for t in range(L, L + STEPS)]

    models: dict[str, list[torch.nn.Module]] = {}
    for arm in ("V1", "B3"):
        models[arm] = []
        for sd in SEEDS:
            m = Model(data, arm).to(dev)
            m.load_state_dict(torch.load(MODEL_DIR / f"{arm}-seed{sd}.pt", map_location=dev))
            models[arm].append(m.eval())

    out: dict = {"test_lied": {"ref_jaccard": {}, "jaccard": {}}}
    for arm in ("V1", "B3"):
        per_prompt = []
        for p in prompts["test_lied"]:
            seed_rows = [run_prompt(p, arm, m, data, S, bidx, pools, draws, pidx, dev)
                         for m in models[arm]]
            per_prompt.append({k: float(np.nanmean([r[k] for r in seed_rows]))
                               for k in ("ref_jaccard", "jaccard")})
        out["test_lied"]["ref_jaccard"][arm] = [r["ref_jaccard"] for r in per_prompt]
        out["test_lied"]["jaccard"][arm] = [r["jaccard"] for r in per_prompt]
        print(arm, "done", flush=True)

    works = [p["work"] for p in prompts["test_lied"]]
    contrast = {}
    for metric in ("ref_jaccard", "jaccard"):
        a = np.array(out["test_lied"][metric]["V1"])
        b = np.array(out["test_lied"][metric]["B3"])
        d = a - b
        keep = ~np.isnan(d)
        contrast[metric] = cluster_bootstrap(d[keep], [w for w, k in zip(works, keep) if k], draws=2000, seed=BOOT_SEED)
        contrast[metric]["mean_v1"] = float(np.nanmean(a))
        contrast[metric]["mean_b3"] = float(np.nanmean(b))
    out["contrast"] = contrast
    (OUT / "reference-jaccard.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    diff, works = load_chain()
    tost = {str(m): tost_p(diff, works, m) for m in (0.25, 0.20, 0.17)}
    (OUT / "tost.json").write_text(json.dumps(tost, indent=1), encoding="utf-8")
    print(json.dumps({"tost": tost, "contrast": contrast}, indent=1))


if __name__ == "__main__":
    main()
