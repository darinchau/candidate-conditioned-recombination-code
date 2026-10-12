# G3 guarded joint objective: V1g (prior-within-guard) vs plain V1 / H-R / R2

Mode full; device cpu; lambdas {'H-R': 1.0, 'H-V': 2.0} (H-R only; V1g has no lambda).
Criteria (pre-registered, plan-gate3-guarded.md @052e969): C1 V1g-V1 trig_lp (full) CI95 lower > 0; C2 V1g-V1 jaccard >= 0 and diatonic >= 0 (paired means); C3 vl_dist/hr_gap/parallels/crossings CI95 upper <= +10% of V1 mean; continuity CI95 lower >= -10%.

Verdict (test_lied, primary): **FAIL** (test_id secondary: FAIL).

## test_id (20 prompts)

| arm | trig_lp full | trig_lp A | trig_lp B | jaccard | diatonic | hr_gap | vl_dist | parallels | crossings | continuity |
|---|---|---|---|---|---|---|---|---|---|---|
| R2 | -7.915 | -8.027 | -7.887 | 0.370 | 0.936 | 0.094 | 6.60 | 0.006 | 0.000 | 0.307 |
| V1 | -7.462 | -7.647 | -7.494 | 0.362 | 0.937 | 0.098 | 9.76 | 0.123 | 0.015 | 0.336 |
| H-R | -5.451 | -5.677 | -5.445 | 0.379 | 0.952 | 0.125 | 6.60 | 0.003 | 0.000 | 0.357 |
| V1g | -5.798 | -6.049 | -5.821 | 0.392 | 0.959 | 0.084 | 8.76 | 0.057 | 0.001 | 0.297 |

| contrast | metric | trigram | mean | 95% CI |
|---|---|---|---|---|
| V1g-V1 | trig_lp | full | +1.663 | [+1.164, +2.147] |
| V1g-V1 | jaccard | full | +0.031 | [+0.011, +0.051] |
| V1g-V1 | diatonic | full | +0.023 | [+0.011, +0.033] |
| V1g-V1 | hr_gap | full | -0.014 | [-0.046, +0.019] |
| V1g-V1 | vl_dist | full | -0.993 | [-1.604, -0.403] |
| V1g-V1 | parallels | full | -0.066 | [-0.099, -0.036] |
| V1g-V1 | crossings | full | -0.014 | [-0.023, -0.004] |
| V1g-V1 | continuity | full | -0.039 | [-0.077, +0.000] |
| V1g-H-R | trig_lp | full | -0.347 | [-0.866, +0.199] |
| V1g-H-R | vl_dist | full | +2.159 | [+1.486, +2.829] |
| V1g-H-R | hr_gap | full | -0.041 | [-0.098, +0.015] |
| V1g-R2 | trig_lp | full | +2.117 | [+1.526, +2.713] |
| V1g-R2 | vl_dist | full | +2.162 | [+1.537, +2.700] |
| V1g-R2 | hr_gap | full | -0.009 | [-0.033, +0.016] |
| H-R-V1 | trig_lp | full | +2.010 | [+1.297, +2.722] |
| H-R-V1 | vl_dist | full | -3.152 | [-3.910, -2.308] |
| H-R-V1 | hr_gap | full | +0.027 | [-0.030, +0.088] |
| V1g-V1 | trig_lp | A | +1.598 | [+1.091, +2.083] |
| V1g-V1 | trig_lp | B | +1.673 | [+1.181, +2.135] |

| criterion | detail | pass |
|---|---|---|
| C1 trig_lp gain | CI95 [+1.164, +2.147] | True |
| C2 independent harmony | jaccard +0.0306, diatonic +0.0225 | True |
| C3 vl_dist | CI95 [-1.604, -0.403] vs tol +0.9755 | True |
| C3 hr_gap | CI95 [-0.046, +0.019] vs tol +0.0098 | False |
| C3 parallels | CI95 [-0.099, -0.036] vs tol +0.0123 | True |
| C3 crossings | CI95 [-0.023, -0.004] vs tol +0.0015 | True |
| C3 continuity | CI95 [-0.077, +0.000] vs tol -0.0336 | False |

**test_id verdict: FAIL** (secondary/informational)

## test_lied (116 prompts)

| arm | trig_lp full | trig_lp A | trig_lp B | jaccard | diatonic | hr_gap | vl_dist | parallels | crossings | continuity |
|---|---|---|---|---|---|---|---|---|---|---|
| R2 | -9.176 | -9.109 | -9.153 | 0.287 | 0.880 | 0.158 | 13.00 | 0.107 | 0.000 | 0.139 |
| V1 | -8.842 | -8.855 | -8.860 | 0.275 | 0.884 | 0.170 | 30.43 | 0.342 | 0.000 | 0.200 |
| H-R | -7.519 | -7.539 | -7.586 | 0.303 | 0.916 | 0.152 | 13.24 | 0.104 | 0.000 | 0.151 |
| V1g | -6.270 | -6.375 | -6.345 | 0.319 | 0.948 | 0.161 | 21.43 | 0.180 | 0.000 | 0.177 |

| contrast | metric | trigram | mean | 95% CI |
|---|---|---|---|---|
| V1g-V1 | trig_lp | full | +2.572 | [+2.369, +2.799] |
| V1g-V1 | jaccard | full | +0.043 | [+0.033, +0.054] |
| V1g-V1 | diatonic | full | +0.064 | [+0.053, +0.074] |
| V1g-V1 | hr_gap | full | -0.009 | [-0.022, +0.003] |
| V1g-V1 | vl_dist | full | -9.000 | [-10.347, -7.714] |
| V1g-V1 | parallels | full | -0.161 | [-0.184, -0.139] |
| V1g-V1 | crossings | full | +0.000 | [+0.000, +0.000] |
| V1g-V1 | continuity | full | -0.023 | [-0.041, -0.005] |
| V1g-H-R | trig_lp | full | +1.249 | [+1.011, +1.501] |
| V1g-H-R | vl_dist | full | +8.189 | [+7.441, +8.998] |
| V1g-H-R | hr_gap | full | +0.009 | [-0.004, +0.022] |
| V1g-R2 | trig_lp | full | +2.906 | [+2.676, +3.150] |
| V1g-R2 | vl_dist | full | +8.431 | [+7.704, +9.258] |
| V1g-R2 | hr_gap | full | +0.002 | [-0.011, +0.015] |
| H-R-V1 | trig_lp | full | +1.323 | [+1.026, +1.627] |
| H-R-V1 | vl_dist | full | -17.189 | [-18.671, -15.768] |
| H-R-V1 | hr_gap | full | -0.018 | [-0.034, -0.002] |
| V1g-V1 | trig_lp | A | +2.481 | [+2.293, +2.690] |
| V1g-V1 | trig_lp | B | +2.515 | [+2.309, +2.743] |

| criterion | detail | pass |
|---|---|---|
| C1 trig_lp gain | CI95 [+2.369, +2.799] | True |
| C2 independent harmony | jaccard +0.0433, diatonic +0.0635 | True |
| C3 vl_dist | CI95 [-10.347, -7.714] vs tol +3.0428 | True |
| C3 hr_gap | CI95 [-0.022, +0.003] vs tol +0.0170 | True |
| C3 parallels | CI95 [-0.184, -0.139] vs tol +0.0342 | True |
| C3 crossings | CI95 [+0.000, +0.000] vs tol +0.0000 | True |
| C3 continuity | CI95 [-0.041, -0.005] vs tol -0.0200 | False |

**test_lied verdict: FAIL** (primary)
