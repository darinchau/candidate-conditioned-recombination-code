"""Standalone re-evaluation of the frozen r2/S2 selection experiment (paper Table I).

Recomputes selection metrics from the shipped checkpoints in models-r2/bn3/S2/ without
retraining, mirroring the evaluation section of run_stage_bn3.main: N3 candidate sets
with frozen CAND_SEEDs, exact-row eligibility restriction, R0/R1/R2 baselines, and the
three seeds per neural arm (C1, B3, V1). Writes <out>/bn-results.json with the same
`kset` layout as the original run, for direct comparison against
frozen/results-r2/bn3/S2/bn-results.json (done automatically by verify.py).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import run_stage_bn3 as bn
from run_stage_a import summarise
from run_stage_b import NEURAL, SEEDS, Data, Model, SuffixR1, metrics, rules_r2, score_neural
from run_stage_b2 import CAND_SEED, EVAL

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results-select-eval")
    args = ap.parse_args()
    out = HERE / args.out
    out.mkdir(parents=True, exist_ok=True)

    bn.TABLE = "r2"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    data = Data(bn.cell_path("S2", False), EVAL)
    cs = {s: bn.candidates_n3(data, s, CAND_SEED[s]) for s in EVAL}
    for s in EVAL:
        ex_mask = cs[s][2].pop("exact")
        bn.restrict_rows(data, s, ex_mask)
        cs[s] = (cs[s][0][ex_mask], cs[s][1][ex_mask], cs[s][2])
    cands = {s: cs[s][0] for s in EVAL}
    types = {s: cs[s][1] for s in EVAL}

    r1 = SuffixR1(data)
    scores: dict[str, dict[str, list[np.ndarray]]] = {s: defaultdict(list) for s in EVAL}
    for s in EVAL:
        scores[s]["R0"].append(np.random.default_rng(1).random(cands[s].shape))
        scores[s]["R1"].append(r1.score(data, s, cands[s]))
        scores[s]["R2"].append(rules_r2(data, s, cands[s]))

    for arm in NEURAL:
        for sd in SEEDS:
            m = Model(data, arm).to(dev)
            m.load_state_dict(torch.load(HERE / "models-r2" / "bn3" / "S2" / f"{arm}-seed{sd}.pt",
                                         map_location=dev))
            m.eval()
            for s in EVAL:
                scores[s][arm].append(score_neural(m, data, s, cands[s], dev))
            del m
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    arms = ["R0", "R1", "R2"] + NEURAL
    results: dict = {"cell": "S2", "table": "r2", "device": str(dev), "arm_order": arms}
    for s in EVAL:
        y, uns = data.ex[s]["y"], data.ex[s]["unseen"]
        results["kset"][s] = {a: {nm: summarise([metrics(sc, cands[s], types[s], y, msk)
                                                 for sc in scores[s][a]])
                                  for nm, msk in (("all", None), ("seen", ~uns), ("unseen", uns))}
                              for a in arms}
    results["wall_seconds"] = round(time.time() - t0, 1)
    (out / "bn-results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {out / 'bn-results.json'} ({results['wall_seconds']}s on {dev})")


if __name__ == "__main__":
    main()
