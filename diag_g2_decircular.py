"""G2 de-circularization: is the harmony-prior gain (and the trig_lp metric) real or count leakage?

Usage: .venv-gpu/Scripts/python.exe diag_g2_decircular.py [--smoke] [--cuda]"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import torch

from diag_common import (HERE, TEST_SPLITS, contrast, half_split, load_models, original_run, run_arm_runs,
                         score_arm, setup, train_seqs, verify_published)
from run_stage_e3e import VL_TOL, WB

PUB = HERE / "results-e3e"
PUB_ARMS = ["R0", "R2", "O-pc", "B3", "V1", "H-R", "H-V", "Original"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    res_dir = HERE / ("results-diag-g2-smoke" if a.smoke else "results-diag-g2")
    res_dir.mkdir(exist_ok=True)
    t0 = time.time()
    dev = torch.device("cuda" if a.cuda and torch.cuda.is_available() else "cpu")
    st = setup(a.smoke)
    ps = {s: st["prompts"][s][:6] if a.smoke else st["prompts"][s] for s in TEST_SPLITS}
    models = load_models(st, dev)
    A, B = half_split(st["data"])
    tri = {"full": WB(train_seqs(st), 3), "A": WB(train_seqs(st, A), 3), "B": WB(train_seqs(st, B), 3)}
    big = {"full": WB(train_seqs(st), 2), "A": WB(train_seqs(st, A), 2), "B": WB(train_seqs(st, B), 2)}
    lam = json.loads((PUB / "chain-results.json").read_text(encoding="utf-8"))["lambda"]
    print(f"halves A={len(A)} B={len(B)} works; lambdas {lam}; device {dev.type}", flush=True)

    # (label, arm, lambda, prior-half); published arms use the full prior at the frozen lambdas
    variants: list[tuple[str, str, float, str | None]] = [
        ("R0", "R0", 0.0, None), ("R2", "R2", 0.0, None), ("O-pc", "O-pc", 0.0, None),
        ("B3", "B3", 0.0, None), ("V1", "V1", 0.0, None), ("Original", "Original", 0.0, None),
        ("H-R", "H-R", lam["H-R"], "full"),
        ("H-RhA", "H-R", lam["H-R"], "A"), ("H-RhB", "H-R", lam["H-R"], "B"),
        ("H-V", "H-V", lam["H-V"], "full"),
        ("H-VhA", "H-V", lam["H-V"], "A"), ("H-VhB", "H-V", lam["H-V"], "B"),
    ]

    out: dict[str, Any] = {"halves": {"A": sorted(A), "B": sorted(B)}, "lambda": lam, "vl_tol": VL_TOL,
                           "device": dev.type, "verify": {}, "per_prompt": {}, "summary": {},
                           "contrasts": {}}
    for split in TEST_SPLITS:
        works = [p["work"] for p in ps[split]]
        rows_all: dict[str, dict[str, list[dict[str, float]]]] = {}
        for label, arm, l, ptag in variants:
            if arm == "Original":
                runs = [[original_run(st, p)] for p in ps[split]]
            else:
                runs = run_arm_runs(st, ps[split], arm, l, big[ptag] if ptag else None, False, models, dev)
            rows_all[label] = {}
            for tname, t in tri.items():
                rows_all[label][tname] = score_arm(st, ps[split], runs, t, False)
                if tname == "full" and not a.smoke and label in PUB_ARMS:
                    out["verify"][f"{split}|{label}"] = verify_published(st, rows_all[label]["full"], split, label)
            print(f"{split}|{label} scored {time.time() - t0:.0f}s", flush=True)
        out["per_prompt"][split] = {"ids": [p["id"] for p in ps[split]], "rows": rows_all}
        out["summary"][split] = {label: {t: {k: float(np.nanmean([r[k] for r in rr]))
                                             for k in ("trig_lp", "jaccard", "vl_dist", "hr_gap", "diatonic")}
                                         for t, rr in tr.items()} for label, tr in rows_all.items()}
        pairs = ([("V1", "R2"), ("B3", "R2"), ("V1", "B3"), ("H-R", "R2"), ("H-V", "V1")]
                 + [("H-RhA", "R2"), ("H-RhB", "R2"), ("H-VhA", "V1"), ("H-VhB", "V1")])
        for xm, ym in pairs:
            for tname in ("full", "A", "B"):
                out["contrasts"][f"{split}|{xm}-{ym}|trig_lp|{tname}"] = contrast(
                    rows_all[xm][tname], rows_all[ym][tname], works, "trig_lp")

    out["wall_seconds"] = time.time() - t0
    (res_dir / "decirc-results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    lines = ["# G2 de-circularization: half-work priors, cross-half trigram scoring", "",
             f"Device {dev.type}; halves A={len(A)}/B={len(B)} works; lambdas {lam}; VL_TOL={VL_TOL}.",
             "Primary readout = cross-half contrasts (hA scored with tri_B, hB with tri_A);",
             "same-side scores diagnose half-specific memorization.", ""]
    for split in TEST_SPLITS:
        lines += [f"## {split} ({len(ps[split])} prompts)", "",
                  "| arm | trig_lp full | trig_lp A | trig_lp B | vl_dist | diatonic | hr_gap |",
                  "|---|---|---|---|---|---|---|"]
        for label, tr in out["summary"][split].items():
            v = tr["full"]
            lines.append(f"| {label} | {v['trig_lp']:.3f} | {tr['A']['trig_lp']:.3f} | {tr['B']['trig_lp']:.3f} | "
                         f"{v['vl_dist']:.2f} | {v['diatonic']:.3f} | {v['hr_gap']:.3f} |")
        lines += ["", "| contrast | trigram | mean | 95% CI |", "|---|---|---|---|"]
        for key, c in out["contrasts"].items():
            s, con, _, tname = key.split("|")
            if s == split and tname in ("full", "A", "B"):
                cross = (con, tname) in (("H-RhA-R2", "B"), ("H-RhB-R2", "A"), ("H-VhA-V1", "B"), ("H-VhB-V1", "A"))
                lines.append(f"| {con} | {tname} | {c['mean']:+.3f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] |"
                             f"{' **cross-half**' if cross else ''}")
        lines.append("")
    (res_dir / "decirc-report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"done {time.time() - t0:.0f}s -> {res_dir}")


if __name__ == "__main__":
    main()
