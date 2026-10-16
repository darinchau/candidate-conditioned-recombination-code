"""G1 structural audit: candidate-pool composition, selection task vs chained generation.

Usage: .venv-gpu/Scripts/python.exe diag_g1_pools.py"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from diag_common import HERE, SEL_SPLITS, TEST_SPLITS, setup
from run_stage_b import K
from run_stage_b2 import CAND_SEED
from run_stage_bn3 import candidates_n3

RES = HERE / "results-diag-g1"


def pool_id_works(data: Any, split: str) -> dict[int, set[str]]:
    P = data.pool[split]
    idw: dict[int, set[str]] = defaultdict(set)
    for r in range(len(P["bid"])):
        idw[int(P["bid"][r])].add(P["work"][r])
    return idw


def audit_rows(data: Any, split: str, cands: np.ndarray, idw: dict[int, set[str]],
               rows_idx: np.ndarray | None = None) -> dict[str, Any]:
    e = data.ex[split]
    y, ew = e["y"], e["work"]
    if rows_idx is None:
        rows_idx = np.arange(len(y))
    n = len(rows_idx)
    present = float(np.mean((cands == y[rows_idx, None]).any(1)))
    own = neg = tgt = 0
    for i, r in enumerate(rows_idx):
        w = ew[int(r)]
        for b in cands[i]:
            if int(b) == int(y[r]):
                continue
            neg += 1
            own += w in idw[int(b)]
            tgt += bool(data.in_train_target[int(b)])
    out = {"rows": n, "K": K, "target_present_rate": round(present, 5),
           "neg_own_work_share": round(own / neg, 5), "neg_train_target_share": round(tgt / neg, 5)}
    return out


def main() -> None:
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    st = setup()
    data = st["data"]
    res: dict[str, Any] = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "e3e_chained_pools": {},
                           "selection_pools": {}}

    # ---- 1. E3e chained pools (target structurally absent) ----
    n = own = tgt = 0
    per_split: dict[str, Any] = {}
    for split in SEL_SPLITS + TEST_SPLITS:
        sn = sown = stgt = 0
        for p in st["prompts"][split]:
            for rows in st["draws"][p["id"]]:
                for j in rows:
                    sn += 1
                    sown += st["pools"][split].work[j] == p["work"]
                    stgt += bool(data.in_train_target[int(st["pools"][split].pb[j])])
        per_split[split] = {"candidates": sn, "own_work": int(sown),
                            "train_target_share": round(stgt / sn, 5)}
        n, own, tgt = n + sn, own + sown, tgt + stgt
    res["e3e_chained_pools"] = {"total_candidates": n, "own_work_total": int(own),
                                "train_target_share": round(tgt / n, 5), "per_split": per_split}

    # ---- 2./3. selection laws ----
    for split in SEL_SPLITS + TEST_SPLITS:
        idw = pool_id_works(data, split)
        t = time.time()
        c2, _, stats2 = data.candidates(split, CAND_SEED[split])
        res["selection_pools"][f"{split}|B2"] = {**audit_rows(data, split, c2, idw, None),
                                                 "draw_stats": {k: v for k, v in stats2.items()}}
        c3, _, stats3 = candidates_n3(data, split, CAND_SEED[split])
        exact = np.flatnonzero(stats3["exact"])
        res["selection_pools"][f"{split}|N3"] = {
            **audit_rows(data, split, c3, idw, None),
            "exact_rows": audit_rows(data, split, c3[exact], idw, exact),
            "fallback_rate": stats3["fallback_rate"]}
        print(f"{split}: B2 + N3 audited in {time.time() - t:.0f}s", flush=True)

    (RES / "pool-audit.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["e3e_chained_pools"], indent=1))
    for k, v in res["selection_pools"].items():
        print(k, {kk: v[kk] for kk in ("rows", "target_present_rate", "neg_own_work_share",
                                       "neg_train_target_share") if kk in v})
    print(f"done {time.time() - t0:.0f}s -> {RES / 'pool-audit.json'}")


if __name__ == "__main__":
    main()
