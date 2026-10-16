"""G3 guarded joint objective (gate 3): V1 + harmony prior as constrained re-ranking.

Usage: .venv-gpu/Scripts/python.exe diag_g3_guarded.py [--smoke] [--cuda]"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import torch

from diag_common import (HERE, METRICS, TEST_SPLITS, contrast, half_split, load_models, run_arm_runs,
                         score_arm, setup, train_seqs, verify_published)
from run_stage_e3e import WB

PUB = HERE / "results-e3e"
CRITERIA = ("C1 V1g-V1 trig_lp (full) CI95 lower > 0",
            "C2 V1g-V1 jaccard >= 0 and diatonic >= 0 (paired means)",
            "C3 vl_dist/hr_gap/parallels/crossings CI95 upper <= +10% of V1 mean; continuity CI95 lower >= -10%")


def evaluate_gate(contrasts: dict[str, Any], v1m: dict[str, float]) -> dict[str, Any]:
    """Pre-registered Pareto gate for one split (plan-gate3-guarded.md)."""
    c1 = contrasts["V1g-V1|trig_lp|full"]
    c2j = contrasts["V1g-V1|jaccard|full"]["mean"]
    c2d = contrasts["V1g-V1|diatonic|full"]["mean"]
    c3: dict[str, Any] = {}
    for k in ("vl_dist", "hr_gap", "parallels", "crossings"):
        c = contrasts[f"V1g-V1|{k}|full"]
        tol = 0.10 * v1m[k]
        c3[k] = {"ci95": c["ci95"], "tol": tol, "pass": bool(c["ci95"][1] <= tol)}
    cc = contrasts["V1g-V1|continuity|full"]
    ctol = -0.10 * v1m["continuity"]
    c3["continuity"] = {"ci95": cc["ci95"], "tol": ctol, "pass": bool(cc["ci95"][0] >= ctol)}
    g: dict[str, Any] = {"c1": {"ci95": c1["ci95"], "pass": bool(c1["ci95"][0] > 0)},
         "c2": {"jaccard_mean": c2j, "diatonic_mean": c2d, "pass": bool(c2j >= 0 and c2d >= 0)},
         "c3": c3}
    g["verdict"] = "PASS" if (g["c1"]["pass"] and g["c2"]["pass"] and all(v["pass"] for v in c3.values())) else "FAIL"
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    res_dir = HERE / ("results-diag-g3-smoke" if a.smoke else "results-diag-g3")
    res_dir.mkdir(exist_ok=True)
    t0 = time.time()
    dev = torch.device("cuda" if a.cuda and torch.cuda.is_available() else "cpu")
    st = setup(a.smoke)
    ps = {s: st["prompts"][s][:6] if a.smoke else st["prompts"][s] for s in TEST_SPLITS}
    models = load_models(st, dev)
    A, B = half_split(st["data"])
    tri = {"full": WB(train_seqs(st), 3), "A": WB(train_seqs(st, A), 3), "B": WB(train_seqs(st, B), 3)}
    big = WB(train_seqs(st), 2)
    lam = json.loads((PUB / "chain-results.json").read_text(encoding="utf-8"))["lambda"]
    print(f"lambdas {lam}; device {dev.type}", flush=True)

    out: dict[str, Any] = {"mode": "smoke" if a.smoke else "full", "lambda": lam, "device": dev.type,
                           "verify": {}, "per_prompt": {}, "summary": {}, "contrasts": {}, "gate": {}}
    for split in TEST_SPLITS:
        works = [p["work"] for p in ps[split]]
        rows_all: dict[str, dict[str, list[dict[str, float]]]] = {}
        for label, arm, l in (("R2", "R2", 0.0), ("V1", "V1", 0.0),
                              ("H-R", "H-R", lam["H-R"]), ("V1g", "V1g", 0.0)):
            runs = run_arm_runs(st, ps[split], arm, l, big, False, models, dev)
            rows_all[label] = {}
            for tname, t in tri.items():
                rows_all[label][tname] = score_arm(st, ps[split], runs, t, False)
                if tname == "full" and not a.smoke and label in ("R2", "V1", "H-R"):
                    out["verify"][f"{split}|{label}"] = verify_published(st, rows_all[label]["full"], split, label)
            print(f"{split}|{label} scored {time.time() - t0:.0f}s", flush=True)
        out["per_prompt"][split] = {"ids": [p["id"] for p in ps[split]], "rows": rows_all}
        out["summary"][split] = {label: {t: {k: float(np.nanmean([r[k] for r in rr])) for k in METRICS}
                                         for t, rr in tr.items()} for label, tr in rows_all.items()}
        cs: dict[str, Any] = {}
        for xm, ym in (("V1g", "V1"), ("V1g", "H-R"), ("V1g", "R2"), ("H-R", "V1")):
            for k in (METRICS if (xm, ym) == ("V1g", "V1") else ("trig_lp", "vl_dist", "hr_gap")):
                cs[f"{xm}-{ym}|{k}|full"] = contrast(rows_all[xm]["full"], rows_all[ym]["full"], works, k)
        for tname in ("A", "B"):
            cs[f"V1g-V1|trig_lp|{tname}"] = contrast(rows_all["V1g"][tname], rows_all["V1"][tname], works, "trig_lp")
        out["contrasts"][split] = cs
        out["gate"][split] = evaluate_gate(cs, out["summary"][split]["V1"]["full"])

    if not a.smoke:
        bad = {k: v for k, v in out["verify"].items() if any(m["max_abs"] > 1e-9 for m in v.values())}
        assert not bad, f"published-anchor verify failed: {list(bad)}"

    out["wall_seconds"] = time.time() - t0
    (res_dir / "gate3-results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    lines = ["# G3 guarded joint objective: V1g (prior-within-guard) vs plain V1 / H-R / R2", "",
             f"Mode {out['mode']}; device {dev.type}; lambdas {lam} (H-R only; V1g has no lambda).",
             f"Criteria (pre-registered, plan-gate3-guarded.md @052e969): {'; '.join(CRITERIA)}.", "",
             f"Verdict (test_lied, primary): **{out['gate']['test_lied']['verdict']}** "
             f"(test_id secondary: {out['gate']['test_id']['verdict']}).", ""]
    for split in TEST_SPLITS:
        lines += [f"## {split} ({len(ps[split])} prompts)", "",
                  "| arm | trig_lp full | trig_lp A | trig_lp B | jaccard | diatonic | hr_gap | vl_dist | parallels | crossings | continuity |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for label, tr in out["summary"][split].items():
            v = tr["full"]
            lines.append(f"| {label} | {v['trig_lp']:.3f} | {tr['A']['trig_lp']:.3f} | {tr['B']['trig_lp']:.3f} | "
                         f"{v['jaccard']:.3f} | {v['diatonic']:.3f} | {v['hr_gap']:.3f} | {v['vl_dist']:.2f} | "
                         f"{v['parallels']:.3f} | {v['crossings']:.3f} | {v['continuity']:.3f} |")
        lines += ["", "| contrast | metric | trigram | mean | 95% CI |", "|---|---|---|---|---|"]
        for key, c in out["contrasts"][split].items():
            con, k, tname = key.rsplit("|", 2)
            lines.append(f"| {con} | {k} | {tname} | {c['mean']:+.3f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] |")
        g = out["gate"][split]
        lines += ["", "| criterion | detail | pass |", "|---|---|---|",
                  f"| C1 trig_lp gain | CI95 [{g['c1']['ci95'][0]:+.3f}, {g['c1']['ci95'][1]:+.3f}] | {g['c1']['pass']} |",
                  f"| C2 independent harmony | jaccard {g['c2']['jaccard_mean']:+.4f}, diatonic {g['c2']['diatonic_mean']:+.4f} | {g['c2']['pass']} |"]
        for k, v in g["c3"].items():
            lines.append(f"| C3 {k} | CI95 [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] vs tol {v['tol']:+.4f} | {v['pass']} |")
        lines += [f"", f"**{split} verdict: {g['verdict']}**"
                  + (" (primary)" if split == "test_lied" else " (secondary/informational)"), ""]
    (res_dir / "gate3-report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"done {time.time() - t0:.0f}s -> {res_dir}; test_lied verdict: {out['gate']['test_lied']['verdict']}")


if __name__ == "__main__":
    main()
