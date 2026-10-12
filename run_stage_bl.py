"""Stage B listening pilot: chained 16-beat recombination with frozen step 2 selectors (plan-stageBL.md, E3).

Every arm sees the same 16 N2-style candidates per step (seeded by prompt and step, independent of history);
the true continuation is not among them. Arms: V1, B3 (models-b2 seed 0), R2 (rule argmax), R0 (random).

Writes results-bl/{STATUS.txt, run.log, bl-results.json, bl-report.md, clip-key-map.json, listening-summary.md,
midi/*.mid, audio/*.wav (if FLUIDSYNTH_BIN and TAIEB_SOUNDFONT are set)}.
Usage: .venv-gpu/Scripts/python.exe run_stage_bl.py [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import time
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_stage_a import cluster_bootstrap
from run_stage_b import K, L, SEED_DATA, Data, Model
from run_stage_b2 import EVAL, env_info

HERE = Path(__file__).resolve().parent
RES = HERE / "results-bl"
SALT = "cc-pilot-BL-v1|"
STEPS = 16
ARMS = ["V1", "B3", "R2", "R0"]
BPM = 72
TPQ = 480


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RES / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def r2_parts(data: Data, p: int, c: int) -> tuple[int, int, int]:
    """(voice-leading distance, parallel 5ths/8ves, crossings) of block p -> c; same logic as run_stage_b.rules_r2."""
    V = data.vals
    last, lastslot = [], []
    for v in range(4):
        seq = [int(x) for x in V[p, v * 4:v * 4 + 4] if x >= 24]
        last.append(seq[-1] if seq else -1)
        nonpad = [int(x) for x in V[p, v * 4:v * 4 + 4] if x != 0]
        lastslot.append(nonpad[-1] if nonpad else 1)
    first = [int(V[c, v * 4]) for v in range(4)]
    dist = sum(12 if (f < 24 or l < 24) else abs(f - l) for f, l in zip(first, last))
    par = 0
    for a in range(4):
        for b in range(a + 1, 4):
            pa, pb_, ca, cb = lastslot[a], lastslot[b], first[a], first[b]
            if min(pa, pb_, ca, cb) < 24:
                continue
            ia, ib = (pa - pb_) % 12, (ca - cb) % 12
            if ia == ib and ia in (0, 7) and ca != pa and cb != pb_ and (ca - pa) * (cb - pb_) > 0:
                par += 1
    cross = sum(1 for a in range(3) if first[a] >= 24 and first[a + 1] >= 24 and first[a] < first[a + 1])
    return dist, par, cross


class Pool:
    """Metric-keyed instance pool of one split (N2 draw: other works, distinct ids, frozen fallback chain)."""

    def __init__(self, data: Data, split: str) -> None:
        P = data.pool[split]
        self.pb, self.pm, self.work = P["bid"], P["met"], P["work"]
        wi = {w: k for k, w in enumerate(sorted(set(self.work)))}
        self.pw = np.array([wi[w] for w in self.work], dtype=np.int64)
        self.wi = wi
        self.keys = [lambda m: (m[0], m[1], m[3]), lambda m: (m[0], m[1]), lambda m: (int(m[1] == 0),), lambda m: ()]
        self.buckets: list[dict[tuple, np.ndarray]] = []
        for kf in self.keys:
            g: dict[tuple, list[int]] = defaultdict(list)
            for r in range(len(self.pb)):
                g[kf(self.pm[r])].append(r)
            self.buckets.append({k: np.array(v) for k, v in g.items()})

    def draw(self, met: list[int], work: str, rng: np.random.Generator) -> list[int]:
        """Return K pool rows (instances) with distinct block ids, none from `work`."""
        w = self.wi.get(work, -1)
        rows: list[int] = []
        seen: set[int] = set()
        for _ in range(K):
            got = None
            for lv, kf in enumerate(self.keys):
                arr = self.buckets[lv].get(kf(met), np.zeros(0, np.int64))
                ok = arr[(self.pw[arr] != w) & ~np.isin(self.pb[arr], list(seen))] if len(arr) else arr
                if len(ok):
                    got = int(ok[rng.integers(len(ok))])
                    break
            assert got is not None
            rows.append(got)
            seen.add(int(self.pb[got]))
        return rows


def piece_ids(data: Data, p: dict) -> list[int]:
    idx = {b: i for i, b in enumerate(data.bids)}
    return [idx[b] for b in p["bids"]]


@torch.no_grad()
def neural_pick(model: Model, hist: list[int], hmet: list[list[int]], qmet: list[int], cands: list[int],
                dev: torch.device) -> int:
    ctx = hist[-L:]
    cm = hmet[-L:]
    pad = L - len(ctx)
    b = {"ctx": torch.tensor([[-1] * pad + ctx], device=dev),
         "cmet": torch.tensor([[[0, 0, 0, 0]] * pad + cm], device=dev),
         "qmet": torch.tensor([qmet], device=dev), "y": torch.zeros(1, dtype=torch.long, device=dev)}
    s = model(b, torch.tensor([cands], device=dev))[0].float().cpu().numpy()
    return int(np.argmax(s))


def generate(data: Data, models: dict[str, Model], pools: dict[str, Pool], p: dict, pi: int, dev: torch.device
             ) -> dict[str, Any]:
    ids = piece_ids(data, p)
    met = [[m[0], m[1], m[2], p["mode"]] for m in p["beats"]]
    out: dict[str, Any] = {"file": p["file"], "work": p["work"], "split": p["split"], "prompt": ids[:L],
                           "original": ids[L:L + STEPS], "arms": {}}
    hist0 = ids[:L]
    pool = pools[p["split"]]
    cand_rows = [pool.draw(met[t], p["work"], np.random.default_rng([SEED_DATA, pi, t])) for t in range(L, L + STEPS)]
    for arm in ARMS:
        hist = list(hist0)
        rows_chosen: list[int] = []
        rng0 = np.random.default_rng([SEED_DATA, pi, 999])
        for k, t in enumerate(range(L, L + STEPS)):
            rows = cand_rows[k]
            cands = [int(pool.pb[r]) for r in rows]
            if arm in models:
                j = neural_pick(models[arm], hist, met[t - L:t], met[t], cands, dev)
            elif arm == "R2":
                sc = [-d - 6 * pa - 3 * cr for d, pa, cr in (r2_parts(data, hist[-1], c) for c in cands)]
                j = int(np.argmax(sc))
            else:
                j = int(rng0.integers(K))
            hist.append(cands[j])
            rows_chosen.append(rows[j])
        out["arms"][arm] = {"blocks": hist[L:], "src_works": [pool.work[r] for r in rows_chosen]}
    out["arms"]["Original"] = {"blocks": ids[L:L + STEPS], "src_works": [p["work"]] * STEPS}
    return out


def seq_metrics(data: Data, prompt_last: int, blocks: list[int], works: list[str]) -> dict[str, float]:
    prev = [prompt_last] + blocks[:-1]
    parts = [r2_parts(data, a, b) for a, b in zip(prev, blocks)]
    cont = [float(works[i] == works[i - 1]) for i in range(1, len(works))]
    return {"vl_dist": float(np.mean([x[0] for x in parts])), "parallels": float(np.mean([x[1] > 0 for x in parts])),
            "crossings": float(np.mean([x[2] > 0 for x in parts])), "continuity": float(np.mean(cont))}


# ----------------------------------------------------------------------------- MIDI / audio
def vlq(n: int) -> bytes:
    b = [n & 0x7F]
    n >>= 7
    while n:
        b.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(b))


def to_midi(data: Data, blocks: list[int], path: Path) -> None:
    """SATB 4-slot grid (simple metre, slot = sixteenth) -> type-0 MIDI, piano, one channel per voice."""
    tick = TPQ // 4
    ev: list[tuple[int, int, bytes]] = []  # (time, order, bytes); note-offs (order 0) before note-ons (1)
    for v in range(4):
        cur = -1
        cells: list[tuple[int, int]] = []
        for b in blocks:
            for s in range(4):
                cells.append((int(data.vals[b, v * 4 + s]), int(data.ons[b, v * 4 + s])))
        for i, (val, on) in enumerate(cells + [(1, 1)]):
            pitch = val if val >= 24 else -1
            if pitch != cur or on:
                if cur >= 0:
                    ev.append((i * tick, 0, bytes([0x80 | v, cur, 0])))
                if pitch >= 0:
                    ev.append((i * tick, 1, bytes([0x90 | v, pitch, 80])))
                cur = pitch
    body = bytearray()
    body += b"\x00\xff\x51\x03" + struct.pack(">I", 60_000_000 // BPM)[1:]
    for v in range(4):
        body += b"\x00" + bytes([0xC0 | v, 0])
    last = 0
    for t, _, msg in sorted(ev, key=lambda x: (x[0], x[1])):
        body += vlq(t - last) + msg
        last = t
    body += vlq(TPQ) + b"\xff\x2f\x00"
    path.write_bytes(b"MThd" + struct.pack(">IHHH", 6, 0, 1, TPQ) + b"MTrk" + struct.pack(">I", len(body)) + bytes(body))


def render(mid: Path, wav: Path) -> dict[str, Any] | None:
    fs, sf = os.environ.get("FLUIDSYNTH_BIN", ""), os.environ.get("TAIEB_SOUNDFONT", "")
    if not (Path(fs).is_file() and Path(sf).is_file()):
        return None
    subprocess.run([fs, "-ni", "-F", str(wav), "-r", "22050", sf, str(mid)], check=True, capture_output=True)
    with wave.open(str(wav), "rb") as a:
        assert a.getframerate() == 22050 and a.getnchannels() == 2 and a.getsampwidth() == 2
        return {"seconds": round(a.getnframes() / 22050, 2), "sha256": hashlib.sha256(wav.read_bytes()).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global RES
    if args.smoke:
        RES = HERE / "results-bl-smoke"
    RES.mkdir(exist_ok=True)
    (RES / "midi").mkdir(exist_ok=True)
    (RES / "audio").mkdir(exist_ok=True)
    (RES / "run.log").write_text("", encoding="utf-8")
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p2 = HERE / "data" / "stageB2" / ("smoke/" if args.smoke else "") / "pieces.json"
    mdir = HERE / ("models-b2-smoke" if args.smoke else "models-b2")
    env = env_info(dev, [p2])
    log(f"env {json.dumps(env)} smoke={args.smoke}")
    data = Data(p2, EVAL)
    models: dict[str, Model] = {}
    for arm in ("V1", "B3"):
        m = Model(data, arm).to(dev)
        m.load_state_dict(torch.load(mdir / f"{arm}-seed0.pt", map_location=dev))
        models[arm] = m.eval()
    pools = {s: Pool(data, s) for s in ("test_id", "test_lied")}

    # eligible prompts: first piece per work, >= L + STEPS beats
    seen_w: set[str] = set()
    elig = []
    for p in data.pieces:
        if p["split"] in ("test_id", "test_lied") and p["work"] not in seen_w and len(p["bids"]) >= L + STEPS:
            seen_w.add(p["work"])
            elig.append(p)
    if args.smoke:
        elig = elig[:6]
    log(f"eligible prompts {len(elig)} (test_id {sum(p['split'] == 'test_id' for p in elig)})")
    gens = [generate(data, models, pools, p, i, dev) for i, p in enumerate(elig)]

    # objective metrics
    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    works: dict[str, list[str]] = defaultdict(list)
    for g in gens:
        for arm, a in g["arms"].items():
            sm = seq_metrics(data, g["prompt"][-1], a["blocks"], a["src_works"])
            for k, v in sm.items():
                per[f"{g['split']}|{arm}"][k].append(v)
        works[g["split"]].append(g["work"])
    summary: dict[str, Any] = {}
    for split in ("test_id", "test_lied"):
        summary[split] = {arm: {k: float(np.mean(v)) for k, v in per[f"{split}|{arm}"].items()}
                          for arm in ARMS + ["Original"]}
        for a, b in (("V1", "B3"), ("V1", "R2"), ("V1", "Original")):
            for k in ("vl_dist", "parallels", "crossings", "continuity"):
                d = np.array(per[f"{split}|{a}"][k]) - np.array(per[f"{split}|{b}"][k])
                summary[split][f"{a}-{b}|{k}"] = cluster_bootstrap(d, works[split], seed=SEED_DATA)
    agree = {f"{a}={b}": float(np.mean([np.mean(np.array(g["arms"][a]["blocks"]) == np.array(g["arms"][b]["blocks"]))
                                         for g in gens])) for a, b in (("V1", "B3"), ("V1", "R2"), ("B3", "R2"))}

    # listening clips
    def ok(p: dict) -> bool:
        return all(b[0] == 4 and b[2] == 0 for b in p["beats"][:L + STEPS])
    gi = {g["file"]: g for g in gens}
    chosen = []
    for split in ("test_id", "test_lied"):
        cand = sorted([p for p in elig if p["split"] == split and ok(p)],
                      key=lambda p: hashlib.sha256((SALT + p["file"]).encode()).hexdigest())
        chosen += cand[:2]
    clip_map: dict[str, Any] = {}
    rng = np.random.default_rng(SEED_DATA)
    for pn, p in enumerate(chosen, 1):
        g = gi[p["file"]]
        order = list(rng.permutation(["Original", "V1", "B3", "R2"]))
        for j, arm in enumerate(order):
            key = f"P{pn}-{'abcd'[j]}"
            mid = RES / "midi" / f"{key}.mid"
            to_midi(data, g["prompt"] + g["arms"][arm]["blocks"], mid)
            rr = render(mid, RES / "audio" / f"{key}.wav")
            clip_map[key] = {"prompt": pn, "arm": arm, "file": p["file"], "split": p["split"],
                             "midi_sha256": hashlib.sha256(mid.read_bytes()).hexdigest(), "audio": rr}
    (RES / "clip-key-map.json").write_text(json.dumps(clip_map, indent=1), encoding="utf-8")
    res = {"env": env, "n_prompts": {s: len(w) for s, w in works.items()}, "objective": summary, "agreement": agree,
           "generations": gens, "listening_prompts": [p["file"] for p in chosen],
           "wall_seconds": round(time.time() - t0, 1)}
    (RES / "bl-results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    f = lambda b: f"{b['mean']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]"  # noqa: E731
    L_ = ["# Stage B listening pilot: objective metrics of chained generation (preliminary)", "",
          f"Commit {env['commit']} (dirty={env['dirty']}); wall {res['wall_seconds']:.0f} s. Prompts {res['n_prompts']}; "
          f"{STEPS} steps from an {L}-beat real prompt; shared N2-style candidates per step, true continuation absent.", "",
          "Per-transition means: voice-leading distance (semitones, SATB summed), share with parallel 5ths/8ves, share "
          "with crossings, share of steps whose block comes from the same source work as the previous step.", ""]
    for split in ("test_id", "test_lied"):
        L_ += [f"## {split}", "", "| arm | VL dist | parallels | crossings | continuity |", "|---|---|---|---|---|"]
        for arm in ARMS + ["Original"]:
            m = summary[split][arm]
            L_.append(f"| {arm} | {m['vl_dist']:.2f} | {m['parallels']:.3f} | {m['crossings']:.3f} | {m['continuity']:.3f} |")
        L_ += ["", "Work-cluster bootstrap:", ""]
        L_ += [f"- {k}: {f(v)}" for k, v in summary[split].items() if "|" in k]
        L_.append("")
    L_ += [f"Arm agreement (share of identical choices): {json.dumps({k: round(v, 3) for k, v in agree.items()})}", "",
           "R2 optimises VL distance, parallels and crossings directly; these metrics are descriptive, not quality."]
    (RES / "bl-report.md").write_text("\n".join(L_) + "\n", encoding="utf-8")
    log(f"DONE in {res['wall_seconds']:.0f} s; agreement {agree}; clips {len(clip_map)}")


if __name__ == "__main__":
    main()
