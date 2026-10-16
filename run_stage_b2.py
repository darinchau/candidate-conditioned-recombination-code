"""Stage B step 2: Lieder in training (frozen plan-stageB2.md).

Usage: .venv-gpu/Scripts/python.exe run_stage_b2.py [--smoke]"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_stage_a import cluster_bootstrap, per_example_correct, summarise
from run_stage_b import (K, L, LRS, NEURAL, SEED_DATA, SEEDS, Data, Model, SuffixR1, kset_metrics, metrics,
                         rules_r2, score_neural, to_batch)

HERE = Path(__file__).resolve().parent
RES = HERE / "results-b2"
MODELS = HERE / "models-b2"
EVAL = ["dev", "test_id", "test_lied", "dev_lied"]
DEVS = ["dev", "dev_lied"]
TESTS = ["test_id", "test_lied"]
# step 1 seeds: dev +0, test_id +1, test_ood +2; dev_lied is new (+3)
CAND_SEED = {"dev": SEED_DATA, "test_id": SEED_DATA + 1, "test_lied": SEED_DATA + 2, "dev_lied": SEED_DATA + 3}
STEP1_SPLIT = {"dev": "dev", "test_id": "test_id", "test_lied": "test_ood"}
LIEDER_PREFIX = "OpenScore/Lieder@38c5db51/"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    RES.mkdir(exist_ok=True)
    with open(RES / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(msg: str) -> None:
    RES.mkdir(exist_ok=True)
    (RES / "STATUS.txt").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n", encoding="utf-8")
    log("STATUS " + msg)


def env_info(dev: torch.device, paths: list[Path]) -> dict:
    def git(*a: str) -> str:
        try:
            return subprocess.check_output(["git", *a], cwd=HERE, text=True).strip()
        except Exception:  # noqa: BLE001
            return "?"
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
            "platform": platform.platform(), "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", ".")),
            "sha256": {p.name if p.parent.name != "stageB" else "stageB/" + p.name:
                       hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def train2(data: Data, arm: str, seed: int, lr: float, dev: torch.device, epochs: int,
           train_cands: list[np.ndarray], dev_cands: dict[str, np.ndarray]) -> tuple[Model, dict]:
    """run_stage_b.train_neural with epoch selection by the mean of dev and dev_lied K=16 NLL."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Model(data, arm).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    e = data.ex["train"]
    n = len(e["y"])
    bs = 256
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * math.ceil(n / bs),
                                                pct_start=0.1)
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
        ms = {s: kset_metrics(score_neural(model, data, s, dev_cands[s], dev), dev_cands[s], data.ex[s]["y"])
              for s in DEVS}
        sel = float(np.mean([ms[s]["nll"] for s in DEVS]))
        hist.append({"epoch": ep, "train_loss": tot / n, "sel_nll": sel,
                     **{f"{s}_{k}": ms[s][k] for s in DEVS for k in ("nll", "top1")}})
        if sel < best[0]:
            best = (sel, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
        if time.time() - t0 > 20 * 60:
            log(f"  {arm} seed{seed} lr{lr}: 20-min budget hit at epoch {ep}")
            break
    model.load_state_dict(best[1])
    return model, {"history": hist, "best_epoch": best[2], "sel_nll": best[0],
                   "train_seconds": round(time.time() - t0, 1)}


def composer(work_file: str) -> str:
    return work_file[len(LIEDER_PREFIX):].split("/")[1] if work_file.startswith(LIEDER_PREFIX) else "-"


def main() -> None:
    global RES, MODELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sm = "smoke/" if args.smoke else ""
    p2 = HERE / "data" / "stageB2" / sm / "pieces.json"
    p1 = HERE / "data" / "stageB" / sm / "pieces.json"
    models1 = HERE / ("models-b-smoke" if args.smoke else "models-b")
    if args.smoke:
        RES, MODELS = HERE / "results-b2-smoke", HERE / "models-b2-smoke"
    frozen_path = RES / "FROZEN-B2.json" if args.smoke else HERE / "FROZEN-B2.json"
    RES.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    (RES / "run.log").write_text("", encoding="utf-8")
    t_start = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    epochs = 2 if args.smoke else 15
    seeds = [0] if args.smoke else SEEDS
    lrs = [1e-3] if args.smoke else LRS
    man = HERE / ("data/stageB2/smoke/split-manifest-B2.json" if args.smoke else "split-manifest-B2.json")
    env = env_info(dev, [p2, p1, man])
    log(f"env {json.dumps(env)}  smoke={args.smoke}")

    # ---- data + identity checks against step 1
    status("loading data (step 2 and step 1)")
    data = Data(p2, EVAL)
    d1 = Data(p1)
    sizes: dict[str, Any] = {s: {"decisions": len(data.ex[s]["y"]), "works": len(set(data.ex[s]["work"])),
                 "unseen": int(data.ex[s]["unseen"].sum())} for s in data.ex}
    tr_src: defaultdict[str, int] = defaultdict(int)
    for p in data.pieces:
        if p["split"] == "train":
            tr_src[p["source"]] += len(p["bids"]) - 1
    sizes["train_by_source"] = dict(tr_src)
    sizes["train_lieder_to_chorale"] = round(tr_src["lieder"] / max(1, tr_src["chorale"]), 3)
    sizes["step1_unseen"] = {s: int(d1.ex[STEP1_SPLIT[s]]["unseen"].sum()) for s in STEP1_SPLIT}
    log(f"blocks {data.N}, vocab {data.V} (step 1 vocab {d1.V}), sizes {json.dumps(sizes)}")
    status("sampling candidate sets + identity check")
    cs = {s: data.candidates(s, CAND_SEED[s]) for s in EVAL}
    cands = {s: cs[s][0] for s in EVAL}
    types = {s: cs[s][1] for s in EVAL}
    ident: dict[str, Any] = {}
    cands1: dict[str, np.ndarray] = {}
    for s, s1 in STEP1_SPLIT.items():
        cc1, t1, _ = d1.candidates(s1, CAND_SEED[s])
        cands1[s] = cc1
        h2 = np.array(data.bids)[cands[s]]
        h1 = np.array(d1.bids)[cc1]
        y_ok = bool((np.array(data.bids)[data.ex[s]["y"]] == np.array(d1.bids)[d1.ex[s1]["y"]]).all())
        ident[s] = {"cands_identical": bool((h2 == h1).all()), "types_identical": bool((types[s] == t1).all()),
                    "targets_identical": y_ok, "works_identical": data.ex[s]["work"] == d1.ex[s1]["work"]}
    log(f"identity {json.dumps(ident)}")
    if not all(all(v.values()) for v in ident.values()):
        status("STOPPED: step 1 identity check failed")
        return
    train_cs = [data.candidates("train", SEED_DATA + 100 + i) for i in range(4)]
    train_cands = [c for c, _, _ in train_cs]
    cand_stats = {"train": [st for _, _, st in train_cs], **{s: cs[s][2] for s in EVAL}}
    log(f"candidate stats {json.dumps(cand_stats)}")

    scores: dict[str, dict[str, list[np.ndarray]]] = {s: defaultdict(list) for s in EVAL}
    r1 = SuffixR1(data)

    def score_baselines(split: str) -> None:
        scores[split]["R0"].append(np.random.default_rng(1).random(cands[split].shape))
        scores[split]["R1"].append(r1.score(data, split, cands[split]))
        scores[split]["R2"].append(rules_r2(data, split, cands[split]))

    status("baselines on dev + dev_lied")
    dev_base: dict[str, dict[str, dict]] = {}
    for s in DEVS:
        score_baselines(s)
        dev_base[s] = {a: metrics(scores[s][a][0], cands[s], types[s], data.ex[s]["y"]) for a in ("R0", "R1", "R2")}
        log(f"{s} baselines " + json.dumps({a: round(m["top1"], 4) for a, m in dev_base[s].items()}))

    # ---- dev-only lr selection on seed 0 (mean dev/dev_lied NLL)
    dev_sel: dict[str, dict] = {}
    seed0: dict[str, Model] = {}
    for arm in NEURAL:
        res: dict[str, dict] = {}
        models: dict[str, Model] = {}
        for lr in lrs:
            status(f"dev selection {arm} lr={lr}")
            model, info = train2(data, arm, 0, lr, dev, epochs, train_cands, {s: cands[s] for s in DEVS})
            ms = {s: metrics(score_neural(model, data, s, cands[s], dev), cands[s], types[s], data.ex[s]["y"])
                  for s in DEVS}
            res[str(lr)] = {**ms, **info}
            models[str(lr)] = model
            log(f"  {arm} lr={lr} sel_nll={info['sel_nll']:.4f} best_ep={info['best_epoch']} "
                + " ".join(f"{s} top1={ms[s]['top1']:.4f} nll={ms[s]['nll']:.4f}" for s in DEVS))
        best_lr = min(res, key=lambda k: res[k]["sel_nll"])
        dev_sel[arm] = {"grid": res, "lr": float(best_lr)}
        seed0[arm] = models[best_lr]
        del models
        torch.cuda.empty_cache()
    best_dev = {s: {a: dev_sel[a]["grid"][str(dev_sel[a]["lr"])][s]["top1"] for a in NEURAL} for s in DEVS}
    gate = {s: bool(best_dev[s]["C1"] <= dev_base[s]["R1"]["top1"] + 0.05
                    and best_dev[s]["C1"] <= dev_base[s]["R2"]["top1"] + 0.05) for s in DEVS}
    gate_ok = all(gate.values())
    dl = dev_base["dev_lied"]
    beats_both = {a: bool(best_dev["dev_lied"][a] > dl["R1"]["top1"] and best_dev["dev_lied"][a] > dl["R2"]["top1"])
                  for a in ("V1", "B3")}
    stop = not (beats_both["V1"] or beats_both["B3"])
    (RES / "stageB2-dev.json").write_text(json.dumps({"baselines": dev_base, "neural": dev_sel, "sizes": sizes,
                                                      "candidate_stats": cand_stats}, indent=1), encoding="utf-8")
    frozen = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "env": env, "epochs": epochs, "K": K, "L": L,
              "seeds": seeds, "lr": {a: dev_sel[a]["lr"] for a in NEURAL}, "identity": ident,
              "dev_top1": {s: {"R0": dev_base[s]["R0"]["top1"], "R1": dev_base[s]["R1"]["top1"],
                               "R2": dev_base[s]["R2"]["top1"], **best_dev[s]} for s in DEVS},
              "validity_gate": {"rule": "C1 <= R1 + 0.05 and C1 <= R2 + 0.05 top-1 on dev and dev_lied (seed 0)",
                                "per_split": gate, "pass": gate_ok},
              "stop_rule": {"rule": "stop if neither V1 nor B3 beats both R1 and R2 on dev_lied",
                            "beats_both": beats_both, "triggered": stop},
              "selection": "lr by mean(dev, dev_lied) K=16 NLL on seed 0; epoch by the same mean",
              "hypotheses": {
                  "HB2-1": "Dgap = (V1-B3)_step2 - (V1-B3)_step1 on test_lied; (a) if CI upper < 0; "
                           "(b) if CI includes 0 and (V1-B3)_step2 CI lower > 0; else inconclusive",
                  "HB2-2": "V1_step2 - R2 test_lied top-1 CI lower > 0",
                  "HB2-3": "V1_step2 - V1_step1 test_lied top-1 CI lower > 0"}}
    frozen_path.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    log(f"FROZEN-B2 written: lr={frozen['lr']} gate={gate} stop={stop} dev_top1={json.dumps(frozen['dev_top1'])}")
    if (not gate_ok or stop) and not args.smoke:
        why = "validity gate failed (C1 leak)" if not gate_ok else "stop rule (no learned arm beats R1 and R2 on dev_lied)"
        status(f"STOPPED before test: {why}")
        write_report(None, frozen, dev_base, dev_sel, sizes, cand_stats, env, t_start)
        return

    # ---- final: seeds x selected lr, score once
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
                model, info = train2(data, arm, seed, dev_sel[arm]["lr"], dev, epochs, train_cands,
                                     {s: cands[s] for s in DEVS})
            train_info[arm].append({"seed": seed, "best_epoch": info["best_epoch"],
                                    "train_seconds": info["train_seconds"]})
            torch.save(model.state_dict(), MODELS / f"{arm}-seed{seed}.pt")
            for s in EVAL:
                scores[s][arm].append(score_neural(model, data, s, cands[s], dev))
            del model
            torch.cuda.empty_cache()

    # ---- step 1 per-example correctness, recomputed from models-b/ with the step 1 Data
    status("step 1 rescoring from models-b")
    s1_arms = ("B3", "V1")
    corr1: dict[str, dict[str, np.ndarray]] = {s: {} for s in STEP1_SPLIT}
    for arm in s1_arms:
        for s, s1 in STEP1_SPLIT.items():
            per = []
            for seed in seeds:
                m = Model(d1, arm).to(dev)
                m.load_state_dict(torch.load(models1 / f"{arm}-seed{seed}.pt", map_location=dev))
                per.append(per_example_correct(score_neural(m, d1, s1, cands1[s], dev), cands1[s], d1.ex[s1]["y"]))
                del m
            corr1[s][arm] = np.mean(per, 0)
    step1_top1 = {s: {a: float(corr1[s][a].mean()) for a in s1_arms} for s in STEP1_SPLIT}
    log(f"step 1 recomputed top-1 {json.dumps(step1_top1)}")

    status("metrics + bootstrap")
    arms = ["R0", "R1", "R2"] + NEURAL
    results: dict[str, Any] = {"kset": {}, "bootstrap": {}, "sizes": sizes, "candidate_stats": cand_stats,
                               "train_info": train_info, "frozen": frozen, "env": env, "step1_top1": step1_top1}
    for s in EVAL:
        e = data.ex[s]
        y, uns, w = e["y"], e["unseen"], e["work"]
        results["kset"][s] = {a: {nm: summarise([metrics(sc, cands[s], types[s], y, msk) for sc in scores[s][a]])
                                  for nm, msk in (("all", None), ("seen", ~uns), ("unseen", uns))} for a in arms}
        c: dict[str, np.ndarray] = {a: np.mean([per_example_correct(sc, cands[s], y) for sc in scores[s][a]], 0) for a in arms}
        diffs: dict[str, np.ndarray] = {"V1-B3": c["V1"] - c["B3"], "V1-R1": c["V1"] - c["R1"],
                                        "V1-R2": c["V1"] - c["R2"], "B3-R2": c["B3"] - c["R2"],
                                        "V1-C1": c["V1"] - c["C1"]}
        if s in STEP1_SPLIT:
            c1: dict[str, np.ndarray] = corr1[s]
            diffs["Dgap"] = (c["V1"] - c["B3"]) - (c1["V1"] - c1["B3"])
            diffs["V1_2-V1_1"] = c["V1"] - c1["V1"]
            diffs["B3_2-B3_1"] = c["B3"] - c1["B3"]
            diffs["(V1-B3)_step1"] = c1["V1"] - c1["B3"]
        bt: dict[str, Any] = {k: cluster_bootstrap(v, w, seed=SEED_DATA) for k, v in diffs.items()}
        slices: dict[str, np.ndarray] = {"unseen": uns, "seen": ~uns}
        if s == "test_lied":
            train_comp = {composer(p["file"]) for p in data.pieces if p["split"] == "train" and p["source"] == "lieder"}
            cin = np.array([composer(x) in train_comp for x in w])
            slices |= {"composer_in_train": cin, "composer_not_in_train": ~cin}
            results["composer_slice_sizes"] = {"in_train": {"decisions": int(cin.sum()),
                                                            "works": len({x for x, k in zip(w, cin) if k})},
                                               "not_in_train": {"decisions": int((~cin).sum()),
                                                                "works": len({x for x, k in zip(w, cin) if not k})}}
        for nm, msk in slices.items():
            if msk.sum() == 0:
                continue
            ww = [w[i] for i in np.where(msk)[0]]
            for k in ("V1-B3", "V1-R2", "Dgap", "V1_2-V1_1", "B3_2-B3_1"):
                if k in diffs:
                    bt[f"{k}|{nm}"] = cluster_bootstrap(diffs[k][msk], ww, seed=SEED_DATA)
            for a in ("R2", "B3", "V1"):
                bt[f"top1_{a}|{nm}"] = {"mean": float(c[a][msk].mean()), "works": len(set(ww)), "n": int(msk.sum())}
        results["bootstrap"][s] = bt
    o = results["bootstrap"]["test_lied"]
    dg, g2 = o["Dgap"], o["V1-B3"]
    if dg["ci95"][1] < 0:
        verdict = "(a) shift robustness: gap shrinks"
    elif dg["ci95"][0] <= 0 <= dg["ci95"][1] and g2["ci95"][0] > 0:
        verdict = "(b) texture advantage: gap persists"
    else:
        verdict = "inconclusive"
    results["H"] = {"HB2-1": {"ci": dg, "gap_step2": g2, "gap_step1": o["(V1-B3)_step1"], "verdict": verdict},
                    "HB2-2": {"ci": o["V1-R2"], "supported": bool(o["V1-R2"]["ci95"][0] > 0)},
                    "HB2-3": {"ci": o["V1_2-V1_1"], "supported": bool(o["V1_2-V1_1"]["ci95"][0] > 0)}}
    results["wall_seconds"] = round(time.time() - t_start, 1)
    results["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else None
    (RES / "stageB2-results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    write_report(results, frozen, dev_base, dev_sel, sizes, cand_stats, env, t_start)
    h = results["H"]
    status(f"DONE in {results['wall_seconds'] / 60:.1f} min; HB2-1={verdict} HB2-2={h['HB2-2']['supported']} "
           f"HB2-3={h['HB2-3']['supported']}")


def write_report(results: dict | None, frozen: dict, dev_base: dict, dev_sel: dict, sizes: dict,
                 cand_stats: dict, env: dict, t0: float) -> None:
    L_ = ["# Stage B step 2 report: Lieder in training (preliminary pilot)", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by run_stage_b2.py. Commit {env['commit']} "
          f"(dirty={env['dirty']}). Device {env['device']}, torch {env['torch']}. "
          f"Wall {(time.time() - t0) / 60:.1f} min.", "",
          "Same task, tokenisation and K=16 candidate recipe as step 1. Train = step 1 chorales + set-disjoint "
          "OpenScore Lieder; dev = chorales, dev_lied = held-out Lieder sets; test_id and test_lied (= step 1 "
          "test_ood) candidate sets are byte-identical to step 1.", "",
          "Sizes: " + ", ".join(f"{s} {v['decisions']} decisions / {v['works']} works / {v['unseen']} unseen"
                                for s, v in sizes.items() if isinstance(v, dict) and "decisions" in v), "",
          f"Train decisions by source: {sizes['train_by_source']} (Lieder:chorale = {sizes['train_lieder_to_chorale']}). "
          f"Step 1 unseen counts on the same eval rows: {sizes['step1_unseen']}.", "",
          "Metric fallback rate: " + ", ".join(f"{s} {cand_stats[s]['metric_fallback_rate']}" for s in EVAL), "",
          "## Dev selection (seed 0; selection = mean of dev and dev_lied NLL)", "",
          "| arm | lr | dev top-1 | dev NLL | dev_lied top-1 | dev_lied NLL |", "|---|---|---|---|---|---|"]
    for a in ("R0", "R1", "R2"):
        d, dl = dev_base["dev"][a], dev_base["dev_lied"][a]
        L_.append(f"| {a} | - | {d['top1']:.3f} | {d['nll']:.3f} | {dl['top1']:.3f} | {dl['nll']:.3f} |")
    for a, v in dev_sel.items():
        for lr, g in v["grid"].items():
            sel = " (selected)" if float(lr) == v["lr"] else ""
            L_.append(f"| {a} | {lr}{sel} | {g['dev']['top1']:.3f} | {g['dev']['nll']:.3f} | "
                      f"{g['dev_lied']['top1']:.3f} | {g['dev_lied']['nll']:.3f} |")
    L_ += ["", f"Validity gate: {frozen['validity_gate']['per_split']} pass={frozen['validity_gate']['pass']}. "
           f"Stop rule triggered: {frozen['stop_rule']['triggered']} ({frozen['stop_rule']['beats_both']}).", ""]
    if results is not None:
        L_ += [f"Step 1 top-1 recomputed from models-b/: {json.dumps(results['step1_top1'])}", ""]
        for s in EVAL:
            L_ += [f"## {s} (mean ± sd over seeds)", "",
                   "| arm | top-1 | top-3 | MRR | NLL | ECE | top-1 seen | top-1 unseen |",
                   "|---|---|---|---|---|---|---|---|"]
            for a, v in results["kset"][s].items():
                al, se, un = v["all"], v["seen"], v["unseen"]
                fs = lambda d: f"{d['top1']:.3f} (n={d['n']})" if d.get("n") else "-"  # noqa: E731
                L_.append(f"| {a} | {al['top1']:.3f}±{al['top1_sd']:.3f} | {al['top3']:.3f} | {al['mrr']:.3f} | "
                          f"{al['nll']:.3f} | {al['ece15']:.3f} | {fs(se)} | {fs(un)} |")
            L_ += ["", "Work-cluster bootstrap (2000 draws, seed-averaged correctness):", ""]
            for k, b in results["bootstrap"][s].items():
                if "ci95" in b:
                    L_.append(f"- {k}: {b['mean']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] over {b['works']} works")
                else:
                    L_.append(f"- {k}: {b['mean']:.4f} (n={b['n']}, {b['works']} works)")
            L_.append("")
        if "composer_slice_sizes" in results:
            L_ += [f"Composer slices (test_lied): {json.dumps(results['composer_slice_sizes'])}", ""]
        h = results["H"]
        f = lambda ci: f"{ci['mean']:+.4f} [{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}]"  # noqa: E731
        L_ += ["## Hypotheses (test_lied)", "",
               f"- HB2-1: Dgap {f(h['HB2-1']['ci'])}; (V1-B3) step 1 {f(h['HB2-1']['gap_step1'])}, "
               f"step 2 {f(h['HB2-1']['gap_step2'])} -> {h['HB2-1']['verdict']}",
               f"- HB2-2: V1_step2 - R2 {f(h['HB2-2']['ci'])}; supported={h['HB2-2']['supported']}",
               f"- HB2-3: V1_step2 - V1_step1 {f(h['HB2-3']['ci'])}; supported={h['HB2-3']['supported']}",
               "", f"Peak GPU memory {results['peak_gpu_mem_gb']} GB.", ""]
    L_ += ["Caveats: preliminary pilot; block continuation is corpus plausibility, not musical quality; no listening "
           "evidence; train size is confounded with texture; composers overlap between train and test_lied."]
    (RES / "stageB2-report.md").write_text("\n".join(L_) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:  # noqa: BLE001
        import traceback
        status(f"FAILED: {type(ex).__name__}: {ex}")
        log(traceback.format_exc())
        raise
