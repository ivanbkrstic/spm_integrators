# Reading the Frequency Response Right — code

Companion code for I. Krstić, "Reading the Frequency Response Right: How to Predict the
Accuracy of Digital Integrators,"

Three independent scripts (Python 3, numpy, scipy; matplotlib only for `--plot`):

| script | what it reproduces |
|---|---|
| `lyons_test.py` | Lyons' test (7 signals, N = 24): errors of the integrals, Table 1 leading terms, shift correction (9), the sin(t) example, best reading instant of the Al-Alaoui integrator |
| `seismic_rules.py` | the seven rules of the earthquake-engineering assessment: worst-case \|ε\| read at offset 0 and at the group delay, the rectangular-rule puzzle, peak vs. RMS errors on Bogdanoff's accelerogram, Eq. (10) |
| `bearing_velocity.py` | real-time velocity from CWRU bearing records 97, 109, 135 (Fig. 2): measured errors and errors from the amplitude spectrum by Eq. (10) |

```
python lyons_test.py [--plot]
python seismic_rules.py [--plot]
python bearing_velocity.py <folder with 97.mat, 109.mat, 135.mat> [--plot]
```

CWRU data: https://engineering.case.edu/bearingdatacenter (drive-end, 48 kHz, 1796 rpm).
