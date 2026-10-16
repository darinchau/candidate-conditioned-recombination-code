# Candidate-Conditioned Recombination — Standalone Reproduction Code

Standalone reproduction package for the AIMG 2026 short paper
**"Harmony and Voice-Leading Tradeoffs in Learned Beat-Block Symbolic Music Mashups"**.

The paper studies whether selection accuracy of learned scorers (a cross-encoder V1 and a
dual encoder B3, against uniform R0, context-free C1, and a voice-leading rule R2) transfers
to chained beat-block generation, and why it does not: target-present training pools,
de-circularized harmony priors, oracle trade-offs, and a guarded objective.

Everything needed to re-derive the paper's numbers is in this repository: code, processed
data, trained checkpoints, and the frozen reference outputs. No download, corpus access,
or retraining is required.

## Contents

| Path | What it is |
|---|---|
| `run_stage_a.py` | Shared library: data containers, models (B3 dual / V1 cross / C1), training, metrics, cluster bootstrap |
| `run_stage_b.py` | Stage B library: K=16 candidate sets, rules (R2), suffix baseline (R1), selection metrics |
| `run_stage_b2.py` | Stage B2 (rest-fixed `r2` table) data plumbing and shared eval splits |
| `run_stage_bl.py` | Pool construction (`Pool`) for chained generation, rule parts, audio render helpers |
| `run_stage_bn3.py` | N3-negative selection experiment (train + eval); `--cell S2 --table r2` = paper Table I |
| `run_stage_e3e.py` | Frozen chained-generation run: 8 arms × 16 steps × 3 seeds over test prompts (paper Table II, Fig. 3) |
| `diag_common.py` | Shared harness: rebuilds the frozen E3e world (data, prompts, pools, draws) and **verifies the pool hash against the published manifest before running** |
| `diag_g1_pools.py` | G1 audit: target-present vs. never-present pools (paper §IV) |
| `diag_g1_inject.py` | G1 causal test: re-inject the original continuation into chained pools (paper Fig. 2, Table III, chorale split Table IV) |
| `diag_g2_decircular.py` | G2: disjoint-half bigram prior, de-circularized rescoring (H-R / H-V) |
| `diag_g3_guarded.py` | G3: guarded objective V1g with pre-registered gate criteria |
| `eval_select.py` | Evaluation-only reproduction of Table I from the shipped checkpoints (no retraining) |
| `verify.py` | Reproduction driver: reruns stages and compares outputs to `frozen/` |
| `plan-stageE3E-frozen.md` | The pre-registered frozen plan for the E3e run (hash-checked by the run manifest) |
| `data/stageB2-r2/pieces.json` | Processed beat-block corpus, rest-fixed r2 table (sha256-checked) |
| `models-r2/bn3/S2/` | Trained checkpoints: `{V1,B3,C1}-seed{0,1,2}.pt` (sha256-checked in the E3e manifest) |
| `frozen/` | Reference outputs the paper's numbers were taken from (never overwritten by `verify.py`) |

## Setup

- Python 3.12, PyTorch 2.9 (original runs used `2.9.1+cu128`), NumPy 2.5
- A GPU is used by `run_stage_e3e.py` and `eval_select.py` if available (all `diag_*`
  stages are CPU). Original hardware: one RTX 5090.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows (else .venv/bin/activate)
pip install -r requirements.txt   # CPU torch by default; for CUDA use the pytorch.org cu128 wheel
python verify.py
```

`verify.py` runs, in order:

| Stage | Command | Runtime (original hardware) | Compares against |
|---|---|---|---|
| `select` | `eval_select.py` | ~2–5 min (GPU) | `frozen/results-r2/bn3/S2/bn-results.json` → Table I |
| `e3e` | `run_stage_e3e.py` | ~3 min (GPU), 159 s originally | `frozen/results-e3e/chain-results.json` → Table II, Fig. 3 |
| `g1pools` | `diag_g1_pools.py` | ~1 min (CPU) | `frozen/results-diag-g1/pool-audit.json` → §IV |
| `g1inject` | `diag_g1_inject.py` | ~2 min (CPU), 81 s originally | `frozen/results-diag-g1/inject-results.json` → Fig. 2, Tables III–IV |
| `g2` | `diag_g2_decircular.py` | ~2 min (CPU) | `frozen/results-diag-g2/decirc-results.json` |
| `g3` | `diag_g3_guarded.py` | ~2 min (CPU) | `frozen/results-diag-g3/gate3-results.json` |

Run a subset with `python verify.py select e3e`, adjust the float tolerance with
`--tol` (default 1e-6; numbers should match exactly on identical hardware, seeds are
frozen throughout). With `python verify.py --train` the driver additionally **retrains**
the Table I scorers from scratch (`run_stage_bn3.py --cell S2 --table r2`, ~35 min on the
original GPU) and compares the fresh `bn-results.json` against the frozen one.

## Paper-number mapping

| Paper artifact | Source |
|---|---|
| Table I (selection top-1, test_id / test_lied) | `frozen/results-r2/bn3/S2/bn-results.json` → `kset.<split>.<arm>.all.top1` |
| Table II (chained metrics on test_lied) | `frozen/results-e3e/chain-results.json` → `summary.test_lied.<arm>` |
| Fig. 2 (pick rates by chain state) | `frozen/results-diag-g1/inject-results.json` → `summary.<split>.<arm>` pick keys |
| Fig. 3 (harmony vs. voice-leading plane) | `frozen/results-e3e/chain-results.json` → `summary.test_lied.<arm>` (trig_lp, vl_dist) |
| Table III (diagnostics with CIs) | `frozen/results-diag-g1/inject-results.json` + `frozen/results-diag-g2/decirc-results.json` + `frozen/results-diag-g3/gate3-results.json` → `contrasts` |
| Table IV (chorale-split injection) | `frozen/results-diag-g1/inject-results.json` → `contrasts` (chorale replication) |
| §IV pool audit numbers | `frozen/results-diag-g1/pool-audit.json` |

## Provenance

- Original runs: 2026-10-08/09/10, Python 3.12.10, torch 2.9.1+cu128, CUDA 12.8,
  one NVIDIA RTX 5090, experiment commit `2e0a50c52dce87c8e2872ca5c8cc4f4f6e61e9ed`.
- `data/stageB2-r2/pieces.json` sha256 `cc0f75e2…d228` matches the value recorded in
  `frozen/results-e3e/run-manifest.json`, as do the six V1/B3 checkpoint hashes.
- The `diag_*` stages re-derive the chained pools from the data file and assert the pool
  hash equals the published manifest's `pool_sha256` before computing anything, so a
  corrupted data or model file cannot silently produce "matching" numbers.
- G4 (listening study) is intentionally not part of this package: it is reported in the
  paper as a pre-registered protocol only.

## Verification status (2026-10-12)

- Data and V1/B3 checkpoint sha256: match `frozen/results-e3e/run-manifest.json`.
- `g1pools`: PASS, exact match to frozen at `--tol 1e-9` (CPU, ~25 s); the pool-hash
  assertion in `diag_common.py` passed.
- `select`, `e3e`, `g1inject`, `g2`, `g3`, `--train`: not yet rerun in this package.

## Repo layout notes

Fresh stage outputs land in `results-e3e/`, `results-diag-g1/`, … and are git-ignored;
the immutable references live under `frozen/`.
