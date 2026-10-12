# Stage E3e plan (frozen 2026-10-09)

Frozen from `plan-stageE3E.md` (draft kept) per `HANDOFF-E3E-FREEZE-RUN-20261009.md`. GPU budget
approved by the researcher on 2026-10-09: cap 0.5 GPU-h, estimate ≤0.25 GPU-h. Runner:
`run_stage_e3e.py`. Outputs go only to `results-e3e/`. CPU audit: `results-e3e-audit/audit.json`.

## 1. Question

Under the E3 chained-recombination protocol, with root-free pc-set harmony metrics:
1. Does candidate-conditioned V1 fit held-out harmony better than B3?
2. Do explicit pc-set harmony priors (H-R, H-V) improve harmony fit at ≤10% `vl_dist` cost?

Per the lit-gap search (revision 2), root-free pc-content metrics are prior art (Yeh et al. 2021;
chord-eval tone-by-tone). E3e uses them as measurement and makes no metric-novelty claim.

## 2. Resolved ambiguities (handoff checklist)

- **pc-set:** the 12-bit mask of pitch classes with `vals ≥ 2` in any of the block's 16 slots. "Mass ≥1
  slot" means sounding in at least one slot. No weighting. An all-rest block is mask 0 and is a normal
  symbol. Duplicate pitches collapse. Pieces are pre-transposed so tonic = pc 0, which the runner asserts
  (`(tonic_pc+shift)%12==0` for all 1,211 pieces). pc-set extraction is asserted deterministic.
- **Trigram metric:** interpolated Witten–Bell over masks with a uniform 1/4096 base and BOS padding,
  fitted on the 925 **train** pieces only. Per step, log2 P(s_t | s_{t−2}, s_{t−1}). History includes
  the prompt's last two blocks. The score is the mean over 16 steps. Higher is better.
- **Jaccard:** per step, against the original continuation's block, averaged. Two empty sets score 1.
- **Diatonic share:** |set ∩ scale| / |set|. Major scale {0,2,4,5,7,9,11}. Minor: natural minor + pc 11.
  Empty sets are excluded (nanmean).
- **Harmonic-rhythm gap:** change rate = the fraction of the 16 transitions (starting from the last prompt
  block) where the mask differs. The gap is |rate(arm) − rate(original)|. Lower is better.
- **H prior:** Witten–Bell bigram from the same train counts, natural-log P(cand mask | previous chosen
  mask).
  - H-R = R2 score (−d − 6·par − 3·cross) + λ·logP.
  - H-V = log_softmax over the K=16 candidates of the V1 logits + λ·logP. This is applied per seed,
    and metrics are then averaged over seeds.
- **λ selection:** grid {0.5, 1, 2, 4}, on dev (26) + dev_lied (46) pooled, 72 prompts.
  - Feasible means pooled mean `vl_dist` ≤ 1.10 × the base arm (R2 or V1) on the same prompts.
  - Choose the feasible λ with the highest pooled mean trig_lp. On an exact tie, take the smaller λ.
  - If none is feasible, the H-arm is reported as not viable and dropped from test.
  - `lambda-selection.json` is written before any test prompt is run.
- **Prompts:** one per work, the first piece in `data.pieces` order with ≥ L+16 blocks. Counts are
  audited: dev 26, dev_lied 46, test_id 20, test_lied 116.
- **Pools:** `run_stage_bl.Pool` per split, K=16. Seed `[20261004, idx, t]`. The idx for test prompts
  equals chain_hd's index, and selection prompts use 10000+i. Pools exclude the prompt's own work, so
  the true continuation is never a candidate. All arms see identical pools. This is audited by
  redrawing, and the pool SHA256 is in the manifest.
- **Checkpoints:** `models-r2/bn3/S2/{V1,B3}-seed{0,1,2}.pt`, used without retraining. SHA256 values
  are in `audit.json`. Seeds 0–2 are averaged per prompt. R0 uses seed `[20261004, idx, 999]`.
- **O-pc (oracle diagnostic):** picks the candidate with the highest Jaccard to the original block. It
  is an upper bound on pool harmony and not a system.

## 3. Arms and contrasts

The arms are R0, R2, B3, V1, O-pc, H-R and H-V, plus the Original as reference. Contrasts are paired
per prompt and use a work-cluster bootstrap (2,000 draws, seed 20261012):
- V1−B3
- H-V−V1
- H-R−R2
- O-pc−V1

**Primary split:** test_lied (116). test_id (20) is secondary.

**Primary metric:** trig_lp. A contrast is "significant" if its 95% CI excludes 0. Jaccard, diatonic
share, hr_gap and the voice-leading metrics are secondary.

**Pre-declared guard (from the smoke-test observation, no rule change):** the H prior and the
trigram metric share train counts, so an H-arm trig_lp gain is partly circular. An H-arm counts as
improving harmony fit only if trig_lp improves **and** neither the hr_gap CI nor the Jaccard CI shows
significant worsening. Otherwise it is reported as a repetition/prior-matching exploit.

## 4. Predictions (directional, test_lied)

- **P1:** V1−B3 trig_lp > 0. Prior evidence for V1's advantage is from proxy metrics only, so this is
  low confidence.
- **P2:** H-V−V1 trig_lp > 0 and passes the guard.
- **P3:** H-R−R2 trig_lp > 0 and passes the guard.
- **P4:** O-pc−V1 Jaccard > 0. This is a sanity check on pool headroom.

## 5. Stopping

- One run with no reruns. Any audit mismatch is a stop.
- No λ-grid extension and no retraining. No listening test in this stage.

## 6. CPU audit results (pre-freeze)

The full values are in `results-e3e-audit/audit.json`.
- **mypy:** clean.
- **Prompt counts:** match 26/46/20/116.
- **Data:** 925 train pieces. 1,962 distinct train pc-sets. Dev unigram OOV rate .0006.
- **Dev trigram:** mean log2 P −6.32.
- **Pools:** 29,583 prompt blocks. Pool SHA256 `27fb0415…`. 28.8% of pool candidates are train-target
  blocks.

**Six-prompt smoke test:** completed on GPU in 11 s, all arms ran. Its results are not committed and
are not evidence.
