"""Reproduction driver for the paper "When Selection Accuracy Does Not Transfer".

Reruns every experiment stage in a fresh checkout and compares the outputs against the
frozen reference artifacts in frozen/. Stages:

  select   eval_select.py: Table I selection metrics from the shipped checkpoints (GPU or CPU)
  e3e      run_stage_e3e.py: frozen chained-generation run (paper Table II + Fig. 2; GPU ~3 min)
  g1pools  diag_g1_pools.py: pool-composition audit (CPU)
  g1inject diag_g1_inject.py: target-injection causal test (paper Fig. 1 + Table III/IV)
  g2       diag_g2_decircular.py: disjoint-half de-circularized prior rescoring
  g3       diag_g3_guarded.py: guarded objective (V1g) gate

Usage:
  python verify.py                 # run all stages, compare to frozen/
  python verify.py select e3e      # run a subset
  python verify.py --tol 1e-9      # float comparison tolerance (default 1e-6)
  python verify.py --train         # additionally retrain from scratch via
                                   # `run_stage_bn3.py --cell S2 --table r2` (~35 min GPU)
                                   # and compare the fresh checkpoints' results

Exit code 0 iff every stage ran and every comparison passed within tolerance.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "frozen"

STAGES = ["select", "e3e", "g1pools", "g1inject", "g2", "g3"]

# Keys ignored at any depth: run metadata that legitimately differs between machines/runs.
SKIP_KEYS = {"written", "wall_seconds", "device", "env", "commit", "dirty", "python", "torch",
             "cuda", "platform", "train_info", "peak_gpu_mem_gb", "run_seconds",
             "train_seconds", "best_epoch", "elapsed"}


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, *(str(c) for c in cmd)], check=True, cwd=HERE)
    print(f"[{time.time() - t0:.0f}s]", flush=True)


def diff(a: Any, b: Any, path: str, tol: float, out: list[str]) -> bool:
    """Compare fresh (a) against frozen (b); append human-readable diffs; return ok."""
    if isinstance(b, dict) and isinstance(a, dict):
        ok = True
        for k in sorted(set(b) | set(a)):
            if k in SKIP_KEYS:
                continue
            if k not in a or k not in b:
                out.append(f"{path}.{k}: present only in {'frozen' if k in b else 'fresh'}")
                ok = False
            else:
                ok = diff(a[k], b[k], f"{path}.{k}", tol, out) and ok
        return ok
    if isinstance(b, list) and isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != frozen {len(b)}")
            return False
        return all(diff(x, y, f"{path}[{i}]", tol, out) for i, (x, y) in enumerate(zip(a, b)))
    if isinstance(b, float) or isinstance(a, float):
        try:
            ok = abs(float(a) - float(b)) <= tol
            if not ok:
                out.append(f"{path}: {a!r} != frozen {b!r}")
            return ok
        except (TypeError, ValueError):
            pass
    if a != b:
        out.append(f"{path}: {a!r} != frozen {b!r}")
        return False
    return True


def compare(stage: str, fresh: Path, frozen: Path, tol: float) -> bool:
    if not fresh.exists():
        print(f"FAIL {stage}: missing output {fresh}")
        return False
    a = json.loads(fresh.read_text(encoding="utf-8"))
    b = json.loads(frozen.read_text(encoding="utf-8"))
    out: list[str] = []
    ok = diff(a, b, "", tol, out)
    n = min(len(out), 10)
    for line in out[:n]:
        print("  " + line)
    if len(out) > n:
        print(f"  ... {len(out) - n} more")
    print(("PASS" if ok else "FAIL") + f" {stage}: {frozen.name} "
          + ("matches frozen" if ok else f"{len(out)} difference(s)"))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stages", nargs="*", choices=STAGES + ["train"], help="subset to run (default: all)")
    ap.add_argument("--tol", type=float, default=1e-6, help="float comparison tolerance")
    args = ap.parse_args()
    todo = args.stages or STAGES
    results: dict[str, bool] = {}

    if "select" in todo:
        run(["eval_select.py"])
        results["select"] = compare("select", HERE / "results-select-eval/bn-results.json",
                                    FROZEN / "results-r2/bn3/S2/bn-results.json", args.tol)

    if "train" in todo:
        run(["run_stage_bn3.py", "--cell", "S2", "--table", "r2"])
        results["train"] = compare("train", HERE / "results-r2/bn3/S2/bn-results.json",
                                   FROZEN / "results-r2/bn3/S2/bn-results.json", args.tol)

    if "e3e" in todo:
        # diag_common verifies reconstructed pools against results-e3e/run-manifest.json;
        # stage the frozen manifest first so the check binds to the published pools.
        pub = HERE / "results-e3e"
        pub.mkdir(exist_ok=True)
        for f in ("run-manifest.json", "chain-results.json"):
            if not (pub / f).exists():
                (pub / f).write_bytes((FROZEN / "results-e3e" / f).read_bytes())
        run(["run_stage_e3e.py"])
        ok = compare("e3e", HERE / "results-e3e/chain-results.json",
                     FROZEN / "results-e3e/chain-results.json", args.tol)
        ok &= compare("e3e-manifest", HERE / "results-e3e/run-manifest.json",
                      FROZEN / "results-e3e/run-manifest.json", args.tol)
        results["e3e"] = ok
    else:
        # Diag stages need the published E3e outputs (pool-hash and chained-metric verification).
        pub = HERE / "results-e3e"
        pub.mkdir(exist_ok=True)
        for f in ("run-manifest.json", "chain-results.json"):
            if not (pub / f).exists():
                (pub / f).write_bytes((FROZEN / "results-e3e" / f).read_bytes())

    if "g1pools" in todo:
        run(["diag_g1_pools.py"])
        results["g1pools"] = compare("g1pools", HERE / "results-diag-g1/pool-audit.json",
                                     FROZEN / "results-diag-g1/pool-audit.json", args.tol)
    if "g1inject" in todo:
        run(["diag_g1_inject.py"])
        results["g1inject"] = compare("g1inject", HERE / "results-diag-g1/inject-results.json",
                                      FROZEN / "results-diag-g1/inject-results.json", args.tol)
    if "g2" in todo:
        run(["diag_g2_decircular.py"])
        results["g2"] = compare("g2", HERE / "results-diag-g2/decirc-results.json",
                                FROZEN / "results-diag-g2/decirc-results.json", args.tol)
    if "g3" in todo:
        run(["diag_g3_guarded.py"])
        results["g3"] = compare("g3", HERE / "results-diag-g3/gate3-results.json",
                                FROZEN / "results-diag-g3/gate3-results.json", args.tol)

    print("\n==== summary ====")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    ok = all(results.values())
    print("ALL PASS" if ok else "SOME STAGES FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
