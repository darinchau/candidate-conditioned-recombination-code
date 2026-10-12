"""MusicXML export of the score examples (paper Fig. 4-9 protocol) for the examples repository.

Rebuilds the frozen E3e world via diag_common.setup() (pool hash verified against
results-e3e/run-manifest.json), chains every arm for each requested prompt with seed-0
models (exactly as render_examples.py does for the LilyPond gallery), and writes one
MusicXML file per prompt x arm: the 8-beat context in grey followed by all 16 generated
beat blocks, one 4/4 measure per block, dashed barlines between generated blocks and a
light-light double barline at the prompt cut. Pitch content is in the key-normalised
frame of the data (tonic C); S/A share staff 1, T/B share staff 2.

Verification:
  --check         round-trips every written file through music21 (parse must succeed and
                  note/rest counts must match) and compares per-voice per-measure event
                  lists (pitch-or-rest, slot count) against data.vals/data.ons for all
                  24 blocks; ties reconstruct the onset flags.
  --check-frozen  compares per-prompt 16-step vl_dist of the deterministic arms
                  (R2, O-pc, H-R) against frozen/results-e3e/chain-results.json.

Usage: python export_examples_xml.py [--split test_lied|test_id] [--prompts N ...]
       [--out xml-examples] [--check] [--check-frozen]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import torch

import diag_common as dc
from render_examples import ARMS, chain_runs
from run_stage_b import L
from run_stage_e3e import STEPS, WB

HERE = Path(__file__).resolve().parent
STEP_NAMES = ("C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B")
STEP_ALTERS = (0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0)
DUR_TYPE = {1: ("16th", 0), 2: ("eighth", 0), 3: ("eighth", 1), 4: ("quarter", 0)}
VOICE_NUM = {0: 1, 1: 2, 2: 5, 3: 6}  # data voice -> MusicXML voice (S/A staff 1, T/B staff 2)
GREY = "#808080"
DOCTYPE = ('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
           '"http://www.musicxml.org/dtds/partwise.dtd">\n')


def block_events(data, b: int, v: int) -> list[list[int]]:
    """Events of voice v in block b as [value, n_slots]; mirrors render_examples.beat_tokens.

    Slots: 0 = PAD (extends the running event), 1 = rest, >=24 = MIDI pitch; `ons` marks onsets.
    A leading PAD slot becomes a rest (events never cross a block boundary; a no-onset repeat
    of the previous pitch becomes a tie instead).
    """
    vals = [int(x) for x in data.vals[b, v * 4:v * 4 + 4]]
    ons = [int(x) for x in data.ons[b, v * 4:v * 4 + 4]]
    ev: list[list[int]] = []
    for k, (x, o) in enumerate(zip(vals, ons)):
        x = 1 if (k == 0 and x == 0) else x
        if k == 0 or (x != 0 and (o == 1 or x != ev[-1][0])):
            ev.append([x, 1])
        else:
            ev[-1][1] += 1
    return ev


def voice_measures(data, blocks: list[int], v: int) -> list[list[dict]]:
    """Per-measure (positional) event lists of voice v, with tie flags at block boundaries."""
    per_m: list[list[dict]] = []
    prev_last: int | None = None
    last_ev: dict | None = None
    for b in blocks:
        ev = block_events(data, b, v)
        first_ties = bool(per_m and prev_last is not None and ev[0][0] == prev_last
                          and int(data.ons[b, v * 4]) == 0)
        cur: list[dict] = []
        for i, (x, n) in enumerate(ev):
            e = {"value": x, "n": n, "tie_start": first_ties and i == 0, "tie_stop": False}
            if last_ev is not None and last_ev["tie_start"]:
                e["tie_stop"] = True
            last_ev = e
            cur.append(e)
        per_m.append(cur)
        prev_last = ev[-1][0] if ev[-1][0] >= 24 else None
    return per_m


def add_note(meas: ET.Element, ev: dict, v: int, grey: bool) -> None:
    voice, staff = VOICE_NUM[v], (1 if v < 2 else 2)
    note = ET.SubElement(meas, "note", {"color": GREY} if grey else {})
    if ev["value"] == 1:
        ET.SubElement(note, "rest")
    else:
        assert ev["value"] >= 24, ev["value"]
        p = ET.SubElement(note, "pitch")
        m = ev["value"]
        ET.SubElement(p, "step").text = STEP_NAMES[m % 12]
        if STEP_ALTERS[m % 12]:
            ET.SubElement(p, "alter").text = str(STEP_ALTERS[m % 12])
        ET.SubElement(p, "octave").text = str(m // 12 - 1)
    ET.SubElement(note, "duration").text = str(ev["n"])
    if ev["tie_stop"]:
        ET.SubElement(note, "tie", {"type": "stop"})
    if ev["tie_start"]:
        ET.SubElement(note, "tie", {"type": "start"})
    ET.SubElement(note, "voice").text = str(voice)
    typ, dots = DUR_TYPE[ev["n"]]
    ET.SubElement(note, "type").text = typ
    for _ in range(dots):
        ET.SubElement(note, "dot")
    ET.SubElement(note, "staff").text = str(staff)
    if ev["tie_stop"] or ev["tie_start"]:
        nots = ET.SubElement(note, "notations")
        if ev["tie_stop"]:
            ET.SubElement(nots, "tied", {"type": "stop"})
        if ev["tie_start"]:
            ET.SubElement(nots, "tied", {"type": "start"})


def musicxml(data, blocks: list[int], ctx: int, meta: dict) -> bytes:
    """One part, two staves, one 4/4 measure per beat block; context measures greyed."""
    per_voice = [voice_measures(data, blocks, v) for v in range(4)]
    root = ET.Element("score-partwise", {"version": "4.0"})
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = \
        f"{meta['work']} -- {meta['arm']} (prompt {meta['prompt_index']}, {meta['split']})"
    misc = ET.SubElement(ET.SubElement(root, "identification"), "miscellaneous")
    for k, val in meta.items():
        ET.SubElement(misc, "miscellaneous-field", {"name": k}).text = str(val)
    ET.SubElement(misc, "miscellaneous-field", {"name": "note"}).text = (
        "Pitches are transposed so the tonic is C (key-normalised frame of the corpus). "
        "Grey measures are the 8-beat prompt context; the double barline marks the cut; "
        "dashed barlines separate generated beat blocks. Learned arms use the seed-0 checkpoint.")
    plist = ET.SubElement(root, "part-list")
    sp = ET.SubElement(plist, "score-part", {"id": "P1"})
    ET.SubElement(sp, "part-name").text = "Piano"
    part = ET.SubElement(root, "part", {"id": "P1"})
    for m_i in range(len(blocks)):
        meas = ET.SubElement(part, "measure", {"number": str(m_i + 1)})
        if m_i == 0:
            att = ET.SubElement(meas, "attributes")
            ET.SubElement(att, "divisions").text = "4"
            key = ET.SubElement(att, "key")
            ET.SubElement(key, "fifths").text = "0"
            tm = ET.SubElement(att, "time")
            ET.SubElement(tm, "beats").text = "4"
            ET.SubElement(tm, "beat-type").text = "4"
            ET.SubElement(att, "staves").text = "2"
            for st_, cl in ((1, "G"), (2, "F")):
                c = ET.SubElement(att, "clef", {"number": str(st_)})
                ET.SubElement(c, "sign").text = cl
                ET.SubElement(c, "line").text = "2"
        grey = m_i < ctx
        for v in range(4):
            if v:
                bak = ET.SubElement(meas, "backup")
                ET.SubElement(bak, "duration").text = "4"
            for ev in per_voice[v][m_i]:
                add_note(meas, ev, v, grey)
        style = "light-light" if m_i == ctx - 1 else "dashed" if m_i < len(blocks) - 1 else "light-heavy"
        ET.SubElement(ET.SubElement(meas, "barline", {"location": "right"}),
                      "bar-style").text = style
    ET.indent(root)
    body = ET.tostring(root, encoding="unicode")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n' + DOCTYPE + body).encode("utf-8")


def check_file(path: Path, data, blocks: list[int]) -> None:
    """Round-trip: music21 must parse the file, and its event content must equal the data.

    The XML tree is compared measure by measure and voice by voice against block_events;
    tie start/stop pairs reconstruct no-onset slot-0 repeats, so vals+ons round-trip exactly.
    """
    import music21

    sc = music21.converter.parse(str(path))
    n_m21 = len(list(sc.recurse().notesAndRests))

    root = ET.parse(path).getroot()
    part = root.find("part")
    assert part is not None
    n_xml = len(part.findall(".//note"))
    per_voice = [voice_measures(data, blocks, v) for v in range(4)]
    n_want = sum(len(m) for v in per_voice for m in v)
    assert n_m21 == n_xml == n_want, f"{path.name}: music21 {n_m21}, xml {n_xml}, data {n_want}"
    for m_i, meas in enumerate(part.findall("measure")):
        by_voice: dict[str, list[ET.Element]] = {}
        for note in meas.findall("note"):
            vel = note.find("voice")
            assert vel is not None and vel.text is not None
            by_voice.setdefault(vel.text, []).append(note)
        for v in range(4):
            got = by_voice.get(str(VOICE_NUM[v]), [])
            want = per_voice[v][m_i]
            assert len(got) == len(want), f"{path.name} m{m_i + 1} v{v}: {len(got)} != {len(want)}"
            for note, ev in zip(got, want):
                del_el = note.find("duration")
                assert del_el is not None and del_el.text is not None
                dur = int(del_el.text)
                pitch = note.find("pitch")
                if pitch is not None:
                    step_el, oct_el = pitch.find("step"), pitch.find("octave")
                    assert step_el is not None and step_el.text is not None
                    assert oct_el is not None and oct_el.text is not None
                    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step_el.text]
                    alt_el = pitch.find("alter")
                    val = (int(oct_el.text) + 1) * 12 + base + \
                        (int(alt_el.text) if alt_el is not None and alt_el.text is not None else 0)
                    assert note.find("rest") is None
                else:
                    assert note.find("rest") is not None
                    val = 1
                assert dur == ev["n"] and val == ev["value"], \
                    f"{path.name} m{m_i + 1} v{v}: xml ({val},{dur}) != data ({ev['value']},{ev['n']})"
                ties = {t.get("type") for t in note.findall("tie")}
                assert (ev["tie_start"] and "start" in ties) or (not ev["tie_start"] and "start" not in ties), \
                    f"{path.name} m{m_i + 1} v{v}: tie start mismatch"
                assert (ev["tie_stop"] and "stop" in ties) or (not ev["tie_stop"] and "stop" not in ties), \
                    f"{path.name} m{m_i + 1} v{v}: tie stop mismatch"


def vl_dist_16(st, ids: list[int], seq: list[int]) -> float:
    import run_stage_bl as bl

    prev = [ids[L - 1]] + seq[:-1]
    return float(np.mean([bl.r2_parts(st["data"], x, y)[0] for x, y in zip(prev, seq)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test_lied", choices=["test_lied", "test_id"])
    ap.add_argument("--prompts", type=int, nargs="+", default=None,
                    help="indices into the split's prompts (default: gallery six for Lieder, all for chorales)")
    ap.add_argument("--out", default="xml-examples")
    ap.add_argument("--check", action="store_true", help="music21 round-trip every written file")
    ap.add_argument("--check-frozen", action="store_true",
                    help="compare 16-step vl_dist of R2/O-pc/H-R against frozen results")
    a = ap.parse_args()
    out = HERE / a.out / ("chorales" if a.split == "test_id" else "lieder")
    out.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cpu")
    st = dc.setup()
    models = dc.load_models(st, dev, seeds=[0])
    big = WB(dc.train_seqs(st), 2)
    lam = json.loads((HERE / "results-e3e/chain-results.json").read_text(encoding="utf-8"))["lambda"]
    frozen = json.loads((HERE / "frozen/results-e3e/chain-results.json").read_text(encoding="utf-8"))
    prompts = st["prompts"][a.split]
    idxs = a.prompts if a.prompts is not None else (
        [0, 25, 27, 40, 45, 48] if a.split == "test_lied" else list(range(len(prompts))))

    n_files = 0
    for pi in idxs:
        p = prompts[pi]
        ids = [st["bidx"][b] for b in p["bids"]]
        runs = chain_runs(st, p, models, big, lam, dev)
        if a.check_frozen:
            rows = frozen["per_prompt"][a.split]["rows"]
            for arm in ("R2", "O-pc", "H-R"):
                want = rows[arm][pi]["vl_dist"]
                got = vl_dist_16(st, ids, runs[arm][:STEPS])
                print(f"prompt {pi:3d} {arm:5s} vl_dist {got:9.4f} frozen {want:9.4f} "
                      f"{'OK' if abs(got - want) < 1e-6 else 'MISMATCH'}")
                assert abs(got - want) < 1e-6, f"chain does not reproduce frozen vl_dist ({pi}, {arm})"
        for arm in ARMS:
            blocks = ids[L - 8:L] + runs[arm][:STEPS]
            meta = {"split": a.split, "prompt_index": pi, "prompt": p["id"], "work": p["work"],
                    "arm": arm, "mode": p["mode"], "tonic_pc": p["tonic_pc"],
                    "generator": "export_examples_xml.py", "context_beats": 8, "steps": STEPS}
            path = out / f"p{pi:03d}-{arm}.xml"
            path.write_bytes(musicxml(st["data"], blocks, 8, meta))
            n_files += 1
            if a.check:
                check_file(path, st["data"], blocks)
                print(f"  ok {path.name}")
    print(f"{n_files} files -> {out}" + ("; all round-trip checks passed" if a.check else ""))


if __name__ == "__main__":
    main()
