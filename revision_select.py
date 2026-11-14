"""Formal paired work-cluster bootstrap test for the V1-B3 selection gap.

This mirrors eval_select.py (frozen r2/S2 selection experiment) but keeps the
per-prompt correctness vector instead of only the aggregate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import run_stage_bn3 as bn
import run_stage_b as b
from run_stage_b2 import CAND_SEED, EVAL

HERE = Path(__file__).resolve().parent
OUT = HERE / "results-revision"
OUT.mkdir(exist_ok=True)


def main() -> None:
    bn.TABLE = "r2"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = b.Data(bn.cell_path("S2", False), EVAL)
    cs = {s: bn.candidates_n3(data, s, CAND_SEED[s]) for s in EVAL}
    for s in EVAL:
        ex_mask = cs[s][2].pop("exact")
        bn.restrict_rows(data, s, ex_mask)
        cs[s] = (cs[s][0][ex_mask], cs[s][1][ex_mask], cs[s][2])
    cands = {s: cs[s][0] for s in EVAL}

    scores = {s: {} for s in EVAL}
    for arm in ("V1", "B3"):
        scores[arm] = {}
        for sd in b.SEEDS:
            m = b.Model(data, arm).to(dev)
            m.load_state_dict(torch.load(HERE / "models-r2" / "bn3" / "S2" / f"{arm}-seed{sd}.pt",
                                         map_location=dev))
            m.eval()
            for s in EVAL:
                scores[arm].setdefault(s, []).append(b.score_neural(m, data, s, cands[s], dev))
            del m
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    # Work-cluster bootstrap of the seed-averaged per-prompt correctness difference.
    from run_stage_a import cluster_bootstrap, per_example_correct

    out = {"cells": {}, "device": str(dev)}
    for s in EVAL:
        y = data.ex[s]["y"]
        corr = {a: np.mean([per_example_correct(sc, cands[s], y) for sc in scores[a][s]], 0)
                for a in ("V1", "B3")}
        w = data.ex[s]["work"]
        d = corr["V1"] - corr["B3"]
        bt = cluster_bootstrap(d, w, draws=2000, seed=b.SEED_DATA)
        # Percentile bootstrap one-sided p for H0: mean <= 0.
        ws = sorted(set(w))
        wi = {x: i for i, x in enumerate(ws)}
        g = np.array([wi[x] for x in w])
        sums = np.bincount(g, weights=d, minlength=len(ws))
        cnts = np.bincount(g, minlength=len(ws)).astype(float)
        rng = np.random.default_rng(b.SEED_DATA)
        idx = rng.integers(0, len(ws), size=(2000, len(ws)))
        bs = sums[idx].sum(1) / cnts[idx].sum(1)
        out["cells"][s] = {
            "mean_v1": float(corr["V1"].mean()),
            "mean_b3": float(corr["B3"].mean()),
            "diff": float(d.mean()),
            "ci95": bt["ci95"],
            "p_one_sided": float((bs <= 0).mean()),
            "n": int(len(d)),
            "works": int(len(ws)),
        }
    (OUT / "selection-test.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
