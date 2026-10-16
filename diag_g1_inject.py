"""G1 causal test (E3e-inject): do the selectors pick the original continuation when it is in the pool?

Usage: .venv-gpu/Scripts/python.exe diag_g1_inject.py [--smoke] [--cuda]"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import torch

from diag_common import (HERE, METRICS, PICK_KEYS, TEST_SPLITS, contrast, load_models, run_arm_runs,
                         score_arm, setup, train_seqs, verify_published)
from run_stage_e3e import WB

RES = HERE / "results-diag-g1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    global RES
    RES = HERE / ("results-diag-g1-smoke" if a.smoke else "results-diag-g1")
    RES.mkdir(exist_ok=True)
    t0 = time.time()
    dev = torch.device("cuda" if a.cuda and torch.cuda.is_available() else "cpu")
    st = setup(a.smoke)
    ps = {s: st["prompts"][s][:6] if a.smoke else st["prompts"][s] for s in TEST_SPLITS}
    models = load_models(st, dev)
    tri_full = WB(train_seqs(st), 3)
    arms = ["R0", "R2", "O-pc", "B3", "V1"]

    # ---- baseline (no inject) + published verification ----
    verify: dict[str, Any] = {}
    for split in TEST_SPLITS:
        for arm in ("R2", "B3", "V1"):
            runs = run_arm_runs(st, ps[split], arm, 0.0, None, False, models, dev)
            rows = score_arm(st, ps[split], runs, tri_full, False)
            if not a.smoke:
                verify[f"{split}|{arm}"] = verify_published(st, rows, split, arm)
    print("verify", json.dumps(verify), f"{time.time() - t0:.0f}s", flush=True)

    # ---- inject runs ----
    inj: dict[str, Any] = {"manifest_ok": True, "baseline_verify": verify, "per_prompt": {}, "summary": {},
                           "contrasts": {}}
    for split in TEST_SPLITS:
        works = [p["work"] for p in ps[split]]
        rows_by_arm = {arm: score_arm(st, ps[split],
                                     run_arm_runs(st, ps[split], arm, 0.0, None, True, models, dev),
                                     tri_full, True) for arm in arms}
        inj["per_prompt"][split] = {"ids": [p["id"] for p in ps[split]], "rows": rows_by_arm}
        inj["summary"][split] = {arm: {k: float(np.nanmean([r[k] for r in rows_by_arm[arm]]))
                                      for k in PICK_KEYS + METRICS} for arm in arms}
        for xm, ym in (("V1", "B3"), ("V1", "R2"), ("R2", "B3")):
            for k in PICK_KEYS + ["trig_lp", "jaccard", "vl_dist", "hr_gap"]:
                inj["contrasts"][f"{split}|{xm}-{ym}|{k}"] = contrast(rows_by_arm[xm], rows_by_arm[ym], works, k)
        print(split, f"{time.time() - t0:.0f}s", flush=True)

    inj["wall_seconds"] = time.time() - t0
    inj["device"] = dev.type
    (RES / "inject-results.json").write_text(json.dumps(inj, indent=1), encoding="utf-8")

    lines = ["# G1 inject diagnostic: original continuation re-injected into E3e pools", "",
             f"Mode {'smoke' if a.smoke else 'full'}; device {dev.type}; arms {arms}; "
             "pick_rate = share of the 16 steps where the arm chose the injected original block "
             "(chance 1/16 = .0625); pick_rate_on = while still on the original path, pick_rate_div = after "
             "first divergence.", ""]
    for split in TEST_SPLITS:
        lines += [f"## {split} ({len(ps[split])} prompts)", "",
                  "| arm | pick_rate | pick_rate_on | pick_rate_div | dup_step_rate | trig_lp | jaccard | vl_dist |",
                  "|---|---|---|---|---|---|---|---|"]
        for arm, v in inj["summary"][split].items():
            lines.append(f"| {arm} | {v['pick_rate']:.3f} | {v['pick_rate_on']:.3f} | {v['pick_rate_div']:.3f} | "
                         f"{v['dup_step_rate']:.3f} | {v['trig_lp']:.3f} | {v['jaccard']:.3f} | {v['vl_dist']:.2f} |")
        lines += ["", "| contrast | metric | mean | 95% CI |", "|---|---|---|---|"]
        for key, c in inj["contrasts"].items():
            s, con, k = key.split("|")
            if s == split:
                lines.append(f"| {con} | {k} | {c['mean']:+.3f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] |")
        lines.append("")
    (RES / "inject-report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"done {time.time() - t0:.0f}s -> {RES}")


if __name__ == "__main__":
    main()
