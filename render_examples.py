"""Score figures for the paper: one prompt of the chosen split continued by each arm, engraved with LilyPond.

Rebuilds the frozen E3e world via diag_common.setup() (pool hash verified against
results-e3e/run-manifest.json), chains the first SHOW steps for each arm with seed-0 models,
and writes per-arm LilyPond fragments plus per-step metrics to <out>/. The fragments are
grand staves (S/A treble, T/B bass) in the key-normalised frame of the data (tonic C).

Usage: python render_examples.py [--prompts N ...] [--show 8] [--out figures-scores] [--engrave]
--engrave calls `lilypond` (2.24) on each .ly to produce cropped PDFs for \\includegraphics.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

import diag_common as dc
import run_stage_bl as bl
from run_stage_b import L
from run_stage_e3e import WB, jaccard

HERE = Path(__file__).resolve().parent
NAMES = ["c", "cis", "d", "ees", "e", "f", "fis", "g", "aes", "a", "bes", "b"]
ARMS = ["Original", "R2", "B3", "V1", "H-R", "V1g", "O-pc"]


def lily_pitch(midi: int) -> str:
    """MIDI -> LilyPond absolute pitch (c' = 60)."""
    octv = midi // 12 - 4
    return NAMES[midi % 12] + ("'" * octv if octv > 0 else "," * -octv)


def beat_tokens(data, blocks: list[int], v: int) -> list[list[str]]:
    """LilyPond tokens of voice v, one list per beat block.

    Slots: 0 = PAD (extends the running event), 1 = rest, >=24 = MIDI pitch; `ons` marks onsets.
    A first slot without onset that repeats the previous beat's last pitch is tied over the barline.
    """
    durs = {1: "16", 2: "8", 3: "8.", 4: "4"}
    out: list[list[str]] = []
    prev_last: int | None = None
    for b in blocks:
        vals = [int(x) for x in data.vals[b, v * 4:v * 4 + 4]]
        ons = [int(x) for x in data.ons[b, v * 4:v * 4 + 4]]
        ev: list[list[int]] = []  # [value, n_slots]
        for k, (x, o) in enumerate(zip(vals, ons)):
            x = 1 if (k == 0 and x == 0) else x
            if k == 0 or (x != 0 and (o == 1 or x != ev[-1][0])):
                ev.append([x, 1])
            else:
                ev[-1][1] += 1
        if out and prev_last is not None and ev[0][0] == prev_last and ons[0] == 0:
            out[-1][-1] += "~"
        out.append([("r" if x < 24 else lily_pitch(x)) + durs[n] for x, n in ev])
        prev_last = ev[-1][0] if ev[-1][0] >= 24 else None
    return out


GREY = (r"\override NoteHead.color = #(x11-color 'grey60) \override Stem.color = #(x11-color 'grey60) "
        r"\override Beam.color = #(x11-color 'grey60) \override Rest.color = #(x11-color 'grey60) "
        r"\override Tie.color = #(x11-color 'grey60) ")
BLACK = r"\revert NoteHead.color \revert Stem.color \revert Beam.color \revert Rest.color \revert Tie.color "


def voice_music(data, blocks: list[int], v: int, ctx: int) -> str:
    """Context beats greyed; dashed bar after every beat block, double bar at the prompt cut."""
    beats = beat_tokens(data, blocks, v)
    parts: list[str] = [GREY]
    for i, toks in enumerate(beats):
        parts.append(" ".join(toks))
        if i == ctx - 1:
            parts.append(r'\bar "||" ' + BLACK)
        elif i < len(beats) - 1:
            parts.append(r'\bar "!"')
    parts.append(r'\bar "|."')
    return " ".join(parts)


def lily_score(data, blocks: list[int], ctx: int, label: str) -> str:
    m = [voice_music(data, blocks, k, ctx) for k in range(4)]
    return (r"\score { \new PianoStaff \with { instrumentName = \markup \bold \sans " + f'"{label}"' + r" } << "
            r"\new Staff { \clef treble \key c \major \cadenzaOn << { \voiceOne " + m[0] + r" } \\ { \voiceTwo "
            + m[1] + r" } >> } "
            r"\new Staff { \clef bass \key c \major \cadenzaOn << { \voiceOne " + m[2] + r" } \\ { \voiceTwo "
            + m[3] + r" } >> } >> "
            r"\layout { \context { \Staff \omit TimeSignature } } }" + "\n")


def lily_doc(scores: list[str]) -> str:
    head = (r'\version "2.24.0"' + "\n" + r"#(set-global-staff-size 14)" + "\n"
            r"\paper { indent = 12\mm short-indent = 12\mm line-width = 176\mm tagline = ##f "
            r"ragged-right = ##f system-system-spacing.basic-distance = #11 "
            r"score-system-spacing.basic-distance = #11 }" + "\n" + r"\header { tagline = ##f }" + "\n")
    return head + "\n".join(scores)


GROUPS = {"scores-learned": ["Original", "B3", "V1"], "scores-tradeoff": ["R2", "O-pc", "H-R", "V1g"]}


def chain_runs(st, p, models, big, lam, dev) -> dict[str, list[int]]:
    ids = [st["bidx"][b] for b in p["bids"]]
    runs: dict[str, list[int]] = {"Original": ids[L:L + dc.STEPS]}
    for arm in ARMS[1:]:
        base = {"V1g": "V1", "H-R": "R2"}.get(arm, arm)
        model = models[base][0] if base in models else None
        runs[arm] = dc.run_prompt_blocks(st, p, arm, float(lam.get(arm, 0.0)), model, big, False, dev)["blocks"]
    return runs


def excerpt_metrics(st, ids: list[int], runs: dict[str, list[int]], show: int, big: WB) -> dict:
    res: dict = {}
    for arm, blocks in runs.items():
        seq = blocks[:show]
        prev = [ids[L - 1]] + seq[:-1]
        parts = [bl.r2_parts(st["data"], x, y) for x, y in zip(prev, seq)]
        res[arm] = {
            "vl_dist": round(float(np.mean([q[0] for q in parts])), 2),
            "parallels": int(sum(q[1] > 0 for q in parts)),
            "crossings": int(sum(q[2] > 0 for q in parts)),
            "jaccard": round(float(np.mean([jaccard(int(st["S"][x]), int(st["S"][o]))
                                            for x, o in zip(seq, runs["Original"][:show])])), 3),
            "bigram_log2p": round(float(np.mean([math.log2(big.prob([int(st["S"][x])], int(st["S"][y])))
                                                 for x, y in zip(prev, seq)])), 2),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", type=int, nargs="+", default=[0], help="indices into the split's prompts")
    ap.add_argument("--split", default="test_lied", choices=["test_lied", "test_id"],
                    help="prompt split to render from")
    ap.add_argument("--show", type=int, default=8, help="generated beats to engrave")
    ap.add_argument("--ctx", type=int, default=2, help="prompt beats shown before the cut")
    ap.add_argument("--out", default="figures-scores")
    ap.add_argument("--engrave", action="store_true")
    a = ap.parse_args()
    out = HERE / a.out
    out.mkdir(exist_ok=True)
    exe = shutil.which("lilypond") or shutil.which("lilypond.exe")
    assert exe or not a.engrave, "lilypond not on PATH"
    # Ghostscript needs a writable temp dir; keep it inside the output tree (sandboxed hosts may deny %TEMP%)
    tmp = out / ".tmp"
    tmp.mkdir(exist_ok=True)
    env = {**os.environ, "TMP": str(tmp), "TEMP": str(tmp), "TMPDIR": str(tmp)}

    dev = torch.device("cpu")
    st = dc.setup()
    models = dc.load_models(st, dev, seeds=[0])
    big = WB(dc.train_seqs(st), 2)
    lam = json.loads((HERE / "results-e3e/chain-results.json").read_text(encoding="utf-8"))["lambda"]

    for pi in a.prompts:
        p = st["prompts"][a.split][pi]
        ids = [st["bidx"][b] for b in p["bids"]]
        runs = chain_runs(st, p, models, big, lam, dev)
        d = out / f"p{pi:03d}"
        d.mkdir(exist_ok=True)
        meta = {"prompt_index": pi, "split": a.split, "prompt": p["id"], "work": p["work"], "mode": p["mode"],
                "tonic_pc": p["tonic_pc"], "beats": p["beats"][L - a.ctx:L + a.show], "show": a.show,
                "ctx": a.ctx, "arms": excerpt_metrics(st, ids, runs, a.show, big)}
        (d / "examples.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        for g, arms in GROUPS.items():
            doc = lily_doc([lily_score(st["data"], ids[L - a.ctx:L] + runs[arm][:a.show], a.ctx, arm)
                            for arm in arms])
            (d / f"{g}.ly").write_text(doc, encoding="utf-8")
            if a.engrave:
                r = subprocess.run([exe, "-dcrop", "-dno-point-and-click", "-dresolution=200", "-fpdf", "-fpng",
                                "-o", g, f"{g}.ly"], cwd=d, capture_output=True, text=True, errors="replace", env=env)
                assert r.returncode == 0, r.stderr[-1500:]
        for f in d.glob("lilypond-tmp-*"):
            f.unlink()
        print(pi, p["work"], json.dumps(meta["arms"]))


if __name__ == "__main__":
    main()
