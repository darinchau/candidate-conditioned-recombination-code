"""Shared harness for the 2026-10-10 G1/G2 diagnostics (post-E3e decision gates).

Reproduces the frozen E3e setup exactly -- data, prompts, seeded pools, draws -- and verifies it
against results-e3e/run-manifest.json (pool_sha256) before anything else runs. Chained runs are
computed once (chosen blocks + inject-pick statistics) and scored separately (metrics depend on
the pc-set trigram supplied at scoring time), so de-circularized re-scoring never repeats model
forwards.

Imported by diag_g1_pools.py / diag_g1_inject.py / diag_g2_decircular.py. No main.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_stage_bl as bl
import run_stage_bn3 as bn
from run_stage_a import cluster_bootstrap
from run_stage_b import L, SEED_DATA, SEEDS, Data, Model
from run_stage_b2 import EVAL
from run_stage_e3e import (BOS, K, LAMBDAS, MODEL_DIR, STEPS, VL_TOL, WB, diatonic, jaccard, logits,
                           pcset)

HERE = Path(__file__).resolve().parent
PUB = HERE / "results-e3e"
SEL_SPLITS = ("dev", "dev_lied")
TEST_SPLITS = ("test_id", "test_lied")
EXPECT_COUNTS = {"dev": 26, "dev_lied": 46, "test_id": 20, "test_lied": 116}
BOOT_SEED = 20261012
HALF_SEED = 20261010
INJECT_SALT = 777  # extra salt for the seeded inject position (G1 causal test)
METRICS = ["trig_lp", "jaccard", "diatonic", "hr_gap", "vl_dist", "parallels", "crossings", "continuity"]
PICK_KEYS = ["pick_rate", "pick_rate_on", "pick_rate_div", "dup_step_rate"]


def setup(smoke: bool = False) -> dict[str, Any]:
    """Frozen E3e world: data, pc-sets, prompts, pools, draws; verified against the published manifest.

    smoke=True keeps full prompts/pools (manifest check stays on) and only flags st["smoke"] so the
    calling script can chain over the first 6 prompts per split (timing/dev only, no evidence).
    """
    bn.TABLE = "r2"
    dpath = bn.cell_path("S2", False)
    data = Data(dpath, EVAL)
    S = np.array([pcset(v) for v in data.vals], dtype=np.int64)
    bidx = {b: i for i, b in enumerate(data.bids)}
    assert all((p["tonic_pc"] + p["shift"]) % 12 == 0 for p in data.pieces), "not key-normalised"
    seen_w: set[str] = set()
    prompts: dict[str, list[dict[str, Any]]] = {s: [] for s in SEL_SPLITS + TEST_SPLITS}
    for p in data.pieces:
        if p["split"] in prompts and p["work"] not in seen_w and len(p["bids"]) >= L + STEPS:
            seen_w.add(p["work"])
            prompts[p["split"]].append(p)
    counts = {s: len(v) for s, v in prompts.items()}
    assert counts == EXPECT_COUNTS, f"prompt counts {counts} != {EXPECT_COUNTS}"
    tlist = [p for p in data.pieces if any(p is q for s in TEST_SPLITS for q in prompts[s])]
    slist = [p for p in data.pieces if any(p is q for s in SEL_SPLITS for q in prompts[s])]
    pidx = {p["id"]: i for i, p in enumerate(tlist)} | {p["id"]: 10000 + i for i, p in enumerate(slist)}
    pools = {s: bl.Pool(data, s) for s in prompts}
    bl.K = K
    draws: dict[str, list[list[int]]] = {}
    for s in prompts:
        for p in prompts[s]:
            met = [[m[0], m[1], m[2], p["mode"]] for m in p["beats"]]
            pi = pidx[p["id"]]
            rows = [pools[s].draw(met[t], p["work"], np.random.default_rng([SEED_DATA, pi, t]))
                    for t in range(L, L + STEPS)]
            for r in rows:
                assert len(r) == K and all(pools[s].work[j] != p["work"] for j in r), "pool leak / size"
            draws[p["id"]] = [[int(j) for j in r] for r in rows]
    man = json.loads((PUB / "run-manifest.json").read_text(encoding="utf-8"))
    got = hashlib.sha256(json.dumps(draws, sort_keys=True).encode()).hexdigest()
    assert got == man["pool_sha256"], "reconstructed pools differ from the published E3e manifest"
    return {"data": data, "S": S, "bidx": bidx, "prompts": prompts, "pidx": pidx, "pools": pools,
            "draws": draws, "manifest": man, "smoke": smoke}


def load_models(st: dict[str, Any], dev: torch.device, seeds: list[int] = SEEDS) -> dict[str, list[Model]]:
    out: dict[str, list[Model]] = {}
    for arm in ("V1", "B3"):
        ms = []
        for sd in seeds:
            m = Model(st["data"], arm).to(dev)
            m.load_state_dict(torch.load(MODEL_DIR / f"{arm}-seed{sd}.pt", map_location=dev))
            ms.append(m.eval())
        out[arm] = ms
    return out


def train_works(data: Data) -> dict[str, str]:
    """work -> source for train pieces (first occurrence)."""
    tw: dict[str, str] = {}
    for p in data.pieces:
        if p["split"] == "train":
            tw.setdefault(p["work"], p["source"])
    return tw


def train_seqs(st: dict[str, Any], works: set[str] | None = None) -> list[list[int]]:
    """Train pc-set sequences, optionally restricted to a set of works."""
    return [[int(st["S"][st["bidx"][b]]) for b in p["bids"]]
            for p in st["data"].pieces if p["split"] == "train" and (works is None or p["work"] in works)]


def half_split(data: Data, seed: int = HALF_SEED) -> tuple[set[str], set[str]]:
    """Split the train works into two halves, stratified by source (chorale/Lieder)."""
    tw = train_works(data)
    rng = np.random.default_rng(seed)
    A: list[str] = []
    B: list[str] = []
    for source in ("chorale", "lieder"):
        ws = sorted(w for w, s in tw.items() if s == source)
        perm = rng.permutation(len(ws))
        for i, idx in enumerate(perm):
            (A if i % 2 == 0 else B).append(ws[idx])
    return set(A), set(B)


def run_prompt_blocks(st: dict[str, Any], p: dict[str, Any], arm: str, lam: float, model: Model | None,
                      big_prior: WB | None, inject: bool, dev: torch.device) -> dict[str, Any]:
    """One chained run over the frozen E3e pools. Returns chosen blocks, source works and pick stats.

    inject=True replaces one seeded-random pool position per step with the ORIGINAL piece's block at
    that position (seed [SEED_DATA, pi, t, INJECT_SALT]); pick stats record which arm picks it.
    """
    s = p["split"]
    ids = [st["bidx"][b] for b in p["bids"]]
    met = [[m[0], m[1], m[2], p["mode"]] for m in p["beats"]]
    hist = ids[:L]
    blocks: list[int] = []
    works: list[str] = []
    rng = np.random.default_rng([SEED_DATA, st["pidx"][p["id"]], 999])
    orig = ids[L:L + STEPS]
    picks = on_steps = on_picks = off_steps = off_picks = dup_steps = 0
    on_track = True
    for k, t in enumerate(range(L, L + STEPS)):
        rows = st["draws"][p["id"]][k]
        cands = [int(st["pools"][s].pb[j]) for j in rows]
        pos = -1
        if inject:
            pos = int(np.random.default_rng([SEED_DATA, st["pidx"][p["id"]], t, INJECT_SALT]).integers(K))
            dup_steps += orig[k] in cands
            cands[pos] = orig[k]
        if arm == "R0":
            j = int(rng.integers(K))
        elif arm == "O-pc":
            j = int(np.argmax([jaccard(int(st["S"][c]), int(st["S"][orig[k]])) for c in cands]))
        elif arm in ("R2", "H-R"):
            sc = np.array([-d - 6 * pa - 3 * cr
                           for d, pa, cr in (bl.r2_parts(st["data"], hist[-1], c) for c in cands)])
            if arm == "H-R":
                assert big_prior is not None and lam
                prev = int(st["S"][hist[-1]])
                sc = sc + lam * np.array([math.log(big_prior.prob([prev], int(st["S"][c]))) for c in cands])
            j = int(np.argmax(sc))
        elif arm == "V1g":
            assert model is not None and big_prior is not None, "V1g needs model + bigram prior"
            sc = logits(model, hist[-L:], met[t - L:t], met[t], cands, st["data"], dev).astype(np.float64)
            j_v1 = int(np.argmax(sc))
            parts = {i: bl.r2_parts(st["data"], hist[-1], c) for i, c in enumerate(cands)}
            prev = int(st["S"][hist[-1]])
            changed = {i: int(int(st["S"][c]) != prev) for i, c in enumerate(cands)}
            safe = [i for i in range(len(cands))
                    if parts[i][0] <= parts[j_v1][0] + 1e-9 and parts[i][1] <= parts[j_v1][1]
                    and parts[i][2] <= parts[j_v1][2] and changed[i] >= changed[j_v1]]
            j = min(safe, key=lambda i: (-math.log(big_prior.prob([prev], int(st["S"][cands[i]]))), -sc[i], i))
        else:
            assert model is not None, f"arm {arm} needs a model"
            sc = logits(model, hist[-L:], met[t - L:t], met[t], cands, st["data"], dev).astype(np.float64)
            if arm == "H-V":
                assert big_prior is not None and lam
                prev = int(st["S"][hist[-1]])
                sc = sc + lam * np.array([math.log(big_prior.prob([prev], int(st["S"][c]))) for c in cands])
            j = int(np.argmax(sc))
        if inject:
            hit = j == pos
            picks += hit
            if on_track:
                on_steps += 1
                on_picks += hit
                on_track = on_track and hit
            else:
                off_steps += 1
                off_picks += hit
        blocks.append(cands[j])
        works.append(st["pools"][s].work[rows[j]])
        hist.append(cands[j])
    return {"blocks": blocks, "works": works, "picks": picks, "on_steps": on_steps, "on_picks": on_picks,
            "off_steps": off_steps, "off_picks": off_picks, "dup_steps": dup_steps}


def original_run(st: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    ids = [st["bidx"][b] for b in p["bids"]]
    return {"blocks": ids[L:L + STEPS], "works": [p["work"]] * STEPS}


def score_prompt(st: dict[str, Any], p: dict[str, Any], run: dict[str, Any], tri_eval: WB,
                 inject: bool) -> dict[str, float]:
    """E3e metrics for one chained run, with the pc-set trigram supplied by the caller."""
    ids = [st["bidx"][b] for b in p["bids"]]
    blocks, works = run["blocks"], run["works"]
    orig = ids[L:L + STEPS]
    r: dict[str, float] = bl.seq_metrics(st["data"], ids[L - 1], blocks, works)
    seq = [int(st["S"][x]) for x in ids[:L]] + [int(st["S"][x]) for x in blocks]
    r["trig_lp"] = float(np.mean([math.log2(tri_eval.prob(seq[i - 2:i], seq[i]))
                                  for i in range(L, L + STEPS)]))
    r["jaccard"] = float(np.mean([jaccard(int(st["S"][x]), int(st["S"][o])) for x, o in zip(blocks, orig)]))
    dv = [diatonic(int(st["S"][x]), p["mode"]) for x in blocks]
    r["diatonic"] = float(np.nanmean(dv)) if not all(math.isnan(x) for x in dv) else float("nan")

    def rate(bs: list[int]) -> float:
        q = [int(st["S"][ids[L - 1]])] + [int(st["S"][x]) for x in bs]
        return float(np.mean([q[i] != q[i - 1] for i in range(1, len(q))]))

    r["hr_gap"] = abs(rate(blocks) - rate(orig))
    out = {k: float(r[k]) for k in METRICS}
    if inject:
        out["pick_rate"] = run["picks"] / STEPS
        out["pick_rate_on"] = run["on_picks"] / run["on_steps"] if run["on_steps"] else float("nan")
        out["pick_rate_div"] = run["off_picks"] / run["off_steps"] if run["off_steps"] else float("nan")
        out["dup_step_rate"] = run["dup_steps"] / STEPS
    return out


def run_arm_runs(st: dict[str, Any], ps: list[dict[str, Any]], arm: str, lam: float, big_prior: WB | None,
                 inject: bool, models: dict[str, list[Model]], dev: torch.device) -> list[list[dict[str, Any]]]:
    """Per-prompt list of per-seed chained runs (neural arms average over seeds at scoring time)."""
    base = {"H-V": "V1", "H-R": "R2", "V1g": "V1"}.get(arm, arm)
    out: list[list[dict[str, Any]]] = []
    for p in ps:
        if base in models:
            out.append([run_prompt_blocks(st, p, arm, lam, m, big_prior, inject, dev) for m in models[base]])
        else:
            out.append([run_prompt_blocks(st, p, arm, lam, None, big_prior, inject, dev)])
    return out


def score_arm(st: dict[str, Any], ps: list[dict[str, Any]], runs: list[list[dict[str, Any]]], tri_eval: WB,
              inject: bool) -> list[dict[str, float]]:
    keys = METRICS + PICK_KEYS if inject else METRICS
    return [{k: float(np.nanmean([score_prompt(st, p, r, tri_eval, inject)[k] for r in rs]))
             for k in keys} for p, rs in zip(ps, runs)]


def contrast(rows_x: list[dict[str, float]], rows_y: list[dict[str, float]], works: list[str], metric: str,
             draws: int = 2000, seed: int = BOOT_SEED) -> dict[str, Any]:
    d = np.array([rx[metric] - ry[metric] for rx, ry in zip(rows_x, rows_y)])
    keep = ~np.isnan(d)
    return cluster_bootstrap(d[keep], [w for w, kk in zip(works, keep) if kk], draws=draws, seed=seed)


def verify_published(st: dict[str, Any], rows: list[dict[str, float]], split: str, arm: str) -> dict[str, Any]:
    """Compare per-prompt rows against the published E3e chain-results.json (same pools/seeds)."""
    pub = json.loads((PUB / "chain-results.json").read_text(encoding="utf-8"))
    prows = pub["per_prompt"][split]["rows"][arm]
    assert len(prows) == len(rows), f"{split}/{arm}: row count {len(rows)} != published {len(prows)}"
    out: dict[str, Any] = {}
    for k in METRICS:
        d = np.array([abs(a[k] - b[k]) for a, b in zip(rows, prows)])
        out[k] = {"max_abs": float(d.max()), "match_1e-9": float(np.mean(d < 1e-9))}
    return out


def jaccard_sets(a: dict[str, float] | None, b: dict[str, float] | None) -> float | None:  # unused helper guard
    return None


def py(v: Any) -> Any:
    """JSON-safe conversion for numpy scalars."""
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    return v
