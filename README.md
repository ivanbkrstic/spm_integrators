# Reading the Frequency Response Right — code

Python code accompanying the manuscript

> I. Krstić, "Reading the Frequency Response Right: How to Predict the Accuracy of Digital Integrators."

The scripts reproduce all numerical results, Table I, Fig. 1 and Fig. 2 of the manuscript.
Equation, figure and table numbers refer to the manuscript. Each script is
self-contained and can be run on its own.

| Script | Reproduces |
|---|---|
| `lyons_test.py` | Lyons' test (seven signals, N = 24): errors of the definite integrals quoted in the text; Table I (leading terms of ε(ω) and orders q, in exact rational arithmetic); "How many samples per period?" (pi/9, about 18 samples per period); the real-sinusoid example (39.5% vs. \|ε\| = 21%); the shift correction (9); the best reading instant of the Al-Alaoui integrator (r = 0.405); the finite-record example (zero padding, extrapolation, Gregory's end weights) |
| `seismic_rules.py` | the seven rules of the earthquake-engineering assessment: worst-case \|ε\| read at offset 0 and at the group delay (Schüssler–Ibler: 200% vs. < 10⁻³); peak versus RMS errors on Bogdanoff's accelerogram; RMS errors from Eq. (10) |
| `figure1.py` | Fig. 1 (`fig1.pdf`, `fig1.png`) |
| `bearing_velocity.py` | Fig. 2 (`fig2.pdf`, `fig2.png`): real-time velocity from CWRU bearing records 97, 109 and 135; measured errors and errors from Eq. (10) (agreement within 0.5%); coefficients of the Chebyshev(5,1) integrator |

## Requirements

Python 3 with NumPy, SciPy and Matplotlib:

```
pip install -r requirements.txt
```

## Usage

```
python lyons_test.py
python seismic_rules.py
python figure1.py
python bearing_velocity.py <folder with 97.mat, 109.mat, 135.mat>
```

## Data

The bearing-vibration records are not included. Download the drive-end records 97 (normal),
109 (inner-race fault) and 135 (outer-race fault), 48 kHz, 1796 rpm, from the
Case Western Reserve University Bearing Data Center:
https://engineering.case.edu/bearingdatacenter
