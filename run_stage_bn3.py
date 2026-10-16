"""Stage B with exchangeable negatives N3 (frozen plan-stageBN3.md, experiment E2b).

Usage: .venv-gpu/Scripts/python.exe run_stage_bn3.py --cell S1|S2|CL|CM [--smoke]"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_stage_b2 as rb2
from run_stage_a import kset_metrics, per_example_correct, summarise
from run_stage_b import K, LRS, NEURAL, SEED_DATA, SEEDS, Data, Model, SuffixR1, metrics, rules_r2, score_neural, to_batch
from run_stage_b2 import CAND_SEED, EVAL, TESTS, env_info

HERE = Path(__file__).resolve().parent
RES = HERE / "results-bn3"
SALT = "cc-pilot-BN3-v1"
C1_MAX = 1.5 / K


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RES / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(msg: str) -> None:
    (RES / "STATUS.txt").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n", encoding="utf-8")
    log("STATUS " + msg)


def bucket_law(ids: np.ndarray, cnt: np.ndarray) -> tuple[np.ndarray, np.ndarray, set[int]] | None:
    """Inclusion probabilities pi = K * p for one bucket (p = instance share). Ids with p > 1/K cannot be
    included with pi <= 1; they are dropped (largest first, renormalising) and recorded. None if < K ids remain."""
    keep = np.ones(len(ids), dtype=bool)
    drop: set[int] = set()
    while keep.sum() >= K:
        p = cnt[keep] / cnt[keep].sum()
        if p.max() <= 1 / K + 1e-12:
            return ids[keep], np.minimum(K * p, 1.0), drop
        j = int(np.argmax(np.where(keep, cnt, -1)))
        keep[j] = False
        drop.add(int(ids[j]))
    return None


def candidates_n3(data: Data, split: str, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """N3 (plan-stageBN3.md): exchangeable, frequency-debiased negatives.

    Bucket = split-pool block instances sharing a metric key; first level whose law contains the target among
    (bpb, beat, compound, mode) -> (compound, mode) -> (compound,) -> (). Within a bucket, a fixed-size-K
    systematic pi-ps design with inclusion probabilities pi_b = K p_b (p = instance share; random order each row)
    is drawn *conditioned on including the target*. Since the target is itself ~ p, P(set, target=b) =
    P(set) * p_b / pi_b = P(set) / K: given the set, every member is equally likely to be the target, so any
    context-free scorer (any function of block id, including frequency and compound flag) has expected top-1 1/K.
    Distinct ids are guaranteed (pi <= 1). Own-work negatives are allowed. Exact on level0 rows; fallback rows
    (target dropped from a dominated fine bucket) are only approximately exchangeable and are counted."""
    rng = np.random.default_rng([seed, int(hashlib.sha256(SALT.encode()).hexdigest()[:8], 16)])
    P = data.pool[split]
    pb, pm = P["bid"], P["met"]
    E = data.ex[split]
    keys = [lambda m: (m[0], m[1], m[2], m[3]), lambda m: (m[2], m[3]), lambda m: (m[2],), lambda m: ()]
    laws: list[dict] = []
    for kf in keys:
        g: defaultdict = defaultdict(list)
        for r in range(len(pb)):
            g[kf(pm[r])].append(int(pb[r]))
        lw = {}
        for key, v in g.items():
            ids, cnt = np.unique(np.array(v, dtype=np.int64), return_counts=True)
            res = bucket_law(ids, cnt.astype(np.float64))
            if res is not None:
                lw[key] = (res[0], res[1], {int(x): i for i, x in enumerate(res[0])})
        laws.append(lw)
    n = len(E["y"])
    out = np.zeros((n, K), dtype=np.int64)
    exact = np.zeros(n, dtype=bool)
    lv_n: Counter = Counter()
    for r in range(n):
        t = int(E["y"][r])
        m = E["qmet"][r]
        law = None
        for lv, kf in enumerate(keys):
            law = laws[lv].get(kf(m))
            if law is not None and t in law[2]:
                break
            law = None
        if law is None:
            raise RuntimeError(f"cannot fill N3 candidate set ({split}, row {r})")
        lv_n[f"level{lv}"] += 1
        exact[r] = lv == 0
        ids, pi, pos = law
        order = rng.permutation(len(ids))
        cum = np.concatenate([[0.0], np.cumsum(pi[order])])
        cum *= K / cum[-1]
        it = int(np.flatnonzero(order == pos[t])[0])
        a0, a1 = cum[it], cum[it + 1]  # target interval, length pi_t <= 1
        # u in [0,1) such that some u + j lands in [a0, a1): uniform over that (possibly wrapped) interval
        u = (a0 + rng.random() * (a1 - a0)) % 1.0
        pts = u + np.arange(K)
        sel = np.clip(np.searchsorted(cum, pts, side="right") - 1, 0, len(ids) - 1)
        row = ids[order[sel]]
        assert len(set(row.tolist())) == K and t in row, (split, r)
        out[r] = row[rng.permutation(K)]
    types = (out != E["y"][:, None]).astype(np.int64)
    return out, types, {"sets": n, "fallback_rate": round(1 - lv_n["level0"] / n, 5),
                        "levels": dict(sorted(lv_n.items())), "exact": exact}


def restrict_rows(data: Data, split: str, keep: np.ndarray) -> None:
    """Keep only scored rows `keep` of data.ex[split] (frozen N3 eligibility: exact level-0 sets)."""
    e = data.ex[split]
    for k in list(e):
        e[k] = [w for w, m in zip(e[k], keep) if m] if k == "work" else e[k][keep]


def train_sel(data: Data, arm: str, seed: int, lr: float, dev: torch.device, epochs: int,
              train_cands: list[np.ndarray], sel_cands: dict[str, np.ndarray]) -> tuple[Model, dict]:
    """run_stage_b2.train2 with an explicit selection-split set (mean K=16 NLL over sel_cands)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Model(data, arm).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    e = data.ex["train"]
    n = len(e["y"])
    bs = 256
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * math.ceil(n / bs), pct_start=0.1)
    rng = np.random.default_rng(seed)
    best: tuple[float, dict[str, torch.Tensor], int] = (float("inf"), {}, -1)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        ce = train_cands[ep % len(train_cands)]
        ypos_all = (ce == e["y"][:, None]).argmax(1)
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            r = perm[i:i + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                s = model(to_batch(e, r, dev), torch.as_tensor(ce[r], device=dev))
                loss = F.cross_entropy(s.float(), torch.as_tensor(ypos_all[r], device=dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(r)
        ms = {s: kset_metrics(score_neural(model, data, s, c, dev), c, data.ex[s]["y"]) for s, c in sel_cands.items()}
        sel = float(np.mean([m["nll"] for m in ms.values()]))
        hist.append({"epoch": ep, "train_loss": tot / n, "sel_nll": sel,
                     **{f"{s}_{k}": ms[s][k] for s in ms for k in ("nll", "top1")}})
        if sel < best[0]:
            best = (sel, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
        if time.time() - t0 > 20 * 60:
            log(f"  {arm} seed{seed} lr{lr}: 20-min budget hit at epoch {ep}")
            break
    model.load_state_dict(best[1])
    return model, {"history": hist, "best_epoch": best[2], "sel_nll": best[0], "train_seconds": round(time.time() - t0, 1)}


TABLE = "r1"  # "r2" = rest-fixed table (data/stageB2-r2, plan-stageR2.md); set via --table or by importers


def cell_path(cell: str, smoke: bool) -> Path:
    sm = "smoke/" if smoke else ""
    if cell == "S2":
        return HERE / "data" / ("stageB2-r2" if TABLE == "r2" else "stageB2") / sm / "pieces.json"
    if cell in ("CL", "CM"):
        return HERE / "data" / "stageBC" / sm / cell / "pieces.json"
    return HERE / "data" / "stageBN" / sm / "S1" / "pieces.json"


def build_s1(smoke: bool) -> Path:
    """S1 = step 2 pieces with every train-Lieder row moved to split 'unused' (block table unchanged)."""
    if TABLE == "r2":
        raise NotImplementedError("S1 has no r2 table (plan-stageR2.md runs S2 only)")
    src = HERE / "data" / "stageB2" / ("smoke/" if smoke else "") / "pieces.json"
    out = cell_path("S1", smoke)
    pieces = json.loads(src.read_text(encoding="utf-8"))
    rows = [({**p, "split": "unused"} if p["split"] == "train" and p["source"] == "lieder" else p) for p in pieces]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows), encoding="utf-8")
    return out


def main() -> None:
    global RES, TABLE
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["S1", "S2", "CL", "CM"], required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--table", choices=["r1", "r2"], default="r1")
    args = ap.parse_args()
    cell = args.cell
    TABLE = args.table
    if TABLE == "r2":
        if cell != "S2":
            ap.error("--table r2 supports --cell S2 only (plan-stageR2.md)")
        RES = HERE / ("results-r2-smoke" if args.smoke else "results-r2") / "bn3" / cell
        models = HERE / ("models-r2-smoke" if args.smoke else "models-r2") / "bn3" / cell
        frozen_path = RES / f"FROZEN-R2-BN3-{cell}.json" if args.smoke else HERE / f"FROZEN-R2-BN3-{cell}.json"
    else:
        RES = HERE / ("results-bn3-smoke" if args.smoke else "results-bn3") / cell
        models = HERE / ("models-bn3-smoke" if args.smoke else "models-bn3") / cell
        frozen_path = RES / f"FROZEN-BN3-{cell}.json" if args.smoke else HERE / f"FROZEN-BN3-{cell}.json"
    rb2.RES = RES
    RES.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    (RES / "run.log").write_text("", encoding="utf-8")
    t_start = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if TABLE == "r2":
        log("table r2 (rest-fixed data/stageB2-r2; plan-stageR2.md)")
    epochs = 2 if args.smoke else 15
    seeds = [0] if args.smoke else SEEDS
    lrs = [1e-3] if args.smoke else LRS
    sel_splits = ["dev"] if cell == "S1" else ["dev", "dev_lied"]
    pc = build_s1(args.smoke) if cell == "S1" else cell_path(cell, args.smoke)
    p2 = cell_path("S2", args.smoke)
    env = env_info(dev, [pc, p2])
    env["cell"] = cell
    env["table"] = TABLE
    log(f"env {json.dumps(env)} smoke={args.smoke}")

    status("loading data")
    data = Data(pc, EVAL + ([] if cell == "S2" else ["unused"]))
    d2 = data if cell == "S2" else Data(p2, EVAL)
    assert data.bids == d2.bids, "block table differs from step 2"
    sizes: dict[str, Any] = {s: {"decisions": len(data.ex[s]["y"]), "works": len(set(data.ex[s]["work"])),
                                 "unseen": int(data.ex[s]["unseen"].sum())} for s in ["train"] + EVAL}
    tr_src: defaultdict[str, int] = defaultdict(int)
    for p in data.pieces:
        if p["split"] == "train":
            tr_src[p["source"]] += len(p["bids"]) - 1
    sizes["train_by_source"] = dict(tr_src)
    sizes["vocab"] = data.V
    log(f"blocks {data.N}, vocab {data.V}, sizes {json.dumps(sizes)}")

    status("sampling N3 candidate sets")
    cs = {s: candidates_n3(data, s, CAND_SEED[s]) for s in EVAL}
    # frozen eligibility: score only rows with an exact (level-0, exchangeable) candidate set
    elig = {s: {"rows_before": int(len(cs[s][0])), "rows_scored": int(cs[s][2]["exact"].sum())} for s in EVAL}
    for s in EVAL:
        ex_mask = cs[s][2].pop("exact")
        restrict_rows(data, s, ex_mask)
        cs[s] = (cs[s][0][ex_mask], cs[s][1][ex_mask], cs[s][2])
    log(f"N3 eligibility {json.dumps(elig)}")
    sizes["eligibility"] = elig
    cands = {s: cs[s][0] for s in EVAL}
    types = {s: cs[s][1] for s in EVAL}
    hb = np.array(data.bids)
    cand_hash = {s: hashlib.sha256(hb[cands[s]].astype("U").tobytes()).hexdigest() for s in EVAL}
    work_hash = {s: hashlib.sha256("|".join(data.ex[s]["work"]).encode()).hexdigest() for s in EVAL}
    log(f"eval candidate-set hashes {json.dumps(cand_hash)}")
    train_cs = [candidates_n3(data, "train", SEED_DATA + 100 + i) for i in range(4)]
    train_cands = [c for c, _, _ in train_cs]
    for _, _, st in train_cs:
        st.pop("exact")
    cand_stats = {"train": [st for _, _, st in train_cs], **{s: cs[s][2] for s in EVAL}}
    log(f"candidate stats {json.dumps(cand_stats)}")

    scores: dict[str, dict[str, list[np.ndarray]]] = {s: defaultdict(list) for s in EVAL}
    r1 = SuffixR1(data)

    def score_baselines(split: str) -> None:
        scores[split]["R0"].append(np.random.default_rng(1).random(cands[split].shape))
        scores[split]["R1"].append(r1.score(data, split, cands[split]))
        scores[split]["R2"].append(rules_r2(data, split, cands[split]))

    devs = ["dev", "dev_lied"]
    status("baselines on dev + dev_lied")
    dev_base: dict[str, dict[str, dict]] = {}
    for s in devs:
        score_baselines(s)
        dev_base[s] = {a: metrics(scores[s][a][0], cands[s], types[s], data.ex[s]["y"]) for a in ("R0", "R1", "R2")}
        log(f"{s} baselines " + json.dumps({a: round(m["top1"], 4) for a, m in dev_base[s].items()}))

    dev_sel: dict[str, dict] = {}
    seed0: dict[str, Model] = {}
    for arm in NEURAL:
        res: dict[str, dict] = {}
        cand_models: dict[str, Model] = {}
        for lr in lrs:
            status(f"dev selection {arm} lr={lr}")
            model, info = train_sel(data, arm, 0, lr, dev, epochs, train_cands, {s: cands[s] for s in sel_splits})
            ms = {s: metrics(score_neural(model, data, s, cands[s], dev), cands[s], types[s], data.ex[s]["y"]) for s in devs}
            res[str(lr)] = {**ms, **info}
            cand_models[str(lr)] = model
            log(f"  {arm} lr={lr} sel_nll={info['sel_nll']:.4f} best_ep={info['best_epoch']} "
                + " ".join(f"{s} top1={ms[s]['top1']:.4f} nll={ms[s]['nll']:.4f}" for s in devs))
        best_lr = min(res, key=lambda k: res[k]["sel_nll"])
        dev_sel[arm] = {"grid": res, "lr": float(best_lr)}
        seed0[arm] = cand_models[best_lr]
        del cand_models
        torch.cuda.empty_cache()
    best_dev = {s: {a: dev_sel[a]["grid"][str(dev_sel[a]["lr"])][s]["top1"] for a in NEURAL} for s in devs}
    gate_splits = sorted(set(sel_splits) | {"dev_lied"})
    gate = {s: bool(best_dev[s]["C1"] <= C1_MAX) for s in gate_splits}
    gate_ok = all(gate.values())
    dl = dev_base["dev_lied"]
    beats_both = {a: bool(best_dev["dev_lied"][a] > dl["R1"]["top1"] and best_dev["dev_lied"][a] > dl["R2"]["top1"])
                  for a in ("V1", "B3")}
    stop = cell != "S1" and not (beats_both["V1"] or beats_both["B3"])
    (RES / "bn-dev.json").write_text(json.dumps({"baselines": dev_base, "neural": dev_sel, "sizes": sizes,
                                                 "candidate_stats": cand_stats}, indent=1), encoding="utf-8")
    frozen = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "cell": cell, "env": env, "epochs": epochs, "seeds": seeds,
              "lr": {a: dev_sel[a]["lr"] for a in NEURAL}, "selection_splits": sel_splits, "sizes": sizes,
              "eval_candidate_sha256": cand_hash, "eval_work_sha256": work_hash,
              "dev_top1": {s: {"R0": dev_base[s]["R0"]["top1"], "R1": dev_base[s]["R1"]["top1"],
                               "R2": dev_base[s]["R2"]["top1"], **best_dev[s]} for s in devs},
              "manipulation_gate": {"rule": f"C1 top-1 <= 1.5/K = {C1_MAX:.4f} on {gate_splits} (seed 0)",
                                    "per_split": gate, "pass": gate_ok},
              "stop_rule": {"rule": "non-S1 cells: stop if neither V1 nor B3 beats both R1 and R2 on dev_lied",
                            "beats_both": beats_both, "triggered": stop}}
    frozen_path.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    log(f"FROZEN written: lr={frozen['lr']} gate={gate} stop={stop} dev_top1={json.dumps(frozen['dev_top1'])}")
    if (not gate_ok or stop) and not args.smoke:
        status("STOPPED before test: " + ("manipulation gate failed" if not gate_ok else "stop rule"))
        return

    status("baselines on test splits")
    for s in TESTS:
        score_baselines(s)
    train_info: dict[str, list] = defaultdict(list)
    for arm in NEURAL:
        for seed in seeds:
            status(f"final {arm} seed={seed}")
            if seed == 0:
                model = seed0.pop(arm)
                info = dev_sel[arm]["grid"][str(dev_sel[arm]["lr"])]
            else:
                model, info = train_sel(data, arm, seed, dev_sel[arm]["lr"], dev, epochs, train_cands,
                                        {s: cands[s] for s in sel_splits})
            train_info[arm].append({"seed": seed, "best_epoch": info["best_epoch"], "train_seconds": info["train_seconds"]})
            torch.save(model.state_dict(), models / f"{arm}-seed{seed}.pt")
            for s in EVAL:
                scores[s][arm].append(score_neural(model, data, s, cands[s], dev))
            del model
            torch.cuda.empty_cache()

    status("metrics")
    arms = ["R0", "R1", "R2"] + NEURAL
    results: dict[str, Any] = {"cell": cell, "kset": {}, "sizes": sizes, "candidate_stats": cand_stats,
                               "train_info": train_info, "frozen": frozen, "env": env}
    corr: dict[str, np.ndarray] = {}
    for s in EVAL:
        y, uns = data.ex[s]["y"], data.ex[s]["unseen"]
        results["kset"][s] = {a: {nm: summarise([metrics(sc, cands[s], types[s], y, msk) for sc in scores[s][a]])
                                  for nm, msk in (("all", None), ("seen", ~uns), ("unseen", uns))} for a in arms}
        for a in arms:
            corr[f"{s}|{a}"] = np.mean([per_example_correct(sc, cands[s], y) for sc in scores[s][a]], 0)
        corr[f"{s}|unseen"] = uns.astype(np.float64)
    np.savez_compressed(RES / "correct.npz", **corr)  # type: ignore[arg-type]  # numpy stub treats **kwds as allow_pickle
    (RES / "works.json").write_text(json.dumps({s: data.ex[s]["work"] for s in EVAL}), encoding="utf-8")
    results["wall_seconds"] = round(time.time() - t_start, 1)
    results["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else None
    (RES / "bn-results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    L_ = [f"# Stage B N3 negatives, cell {cell} (preliminary pilot)", "",
          f"Commit {env['commit']} (dirty={env['dirty']}), {env['device']}, wall {results['wall_seconds'] / 60:.1f} min, "
          f"peak {results['peak_gpu_mem_gb']} GB. Train by source {sizes['train_by_source']}. lr {frozen['lr']}. "
          f"Manipulation gate {gate} pass={gate_ok}.", ""]
    for s in EVAL:
        L_ += [f"## {s}", "", "| arm | top-1 | NLL | ECE | top-1 unseen |", "|---|---|---|---|---|"]
        for a, v in results["kset"][s].items():
            al, un = v["all"], v["unseen"]
            L_.append(f"| {a} | {al['top1']:.3f}±{al['top1_sd']:.3f} | {al['nll']:.3f} | {al['ece15']:.3f} | "
                      + (f"{un['top1']:.3f} (n={un['n']})" if un.get("n") else "-") + " |")
        L_.append("")
    L_.append("Cross-cell contrasts: analyse_bn3.py. Caveat: K-set proxy under N3 negatives, exact level-0 rows only; not comparable to N1/N2 numbers.")
    (RES / "bn-report.md").write_text("\n".join(L_) + "\n", encoding="utf-8")
    status(f"DONE in {results['wall_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:  # noqa: BLE001
        import traceback
        status(f"FAILED: {type(ex).__name__}: {ex}")
        log(traceback.format_exc())
        raise
