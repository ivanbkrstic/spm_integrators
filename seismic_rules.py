"""Seven integration rules of an earthquake-engineering assessment, read at the right instant.

Companion code for I. Krstic, "Reading the Frequency Response Right: How to Predict
the Accuracy of Digital Integrators".

Assessment: J. Arias-Trujillo, R. Blazquez, S. Lopez-Querol, "Frequency-domain assessment
of integration schemes for earthquake engineering problems," Math. Probl. Eng., 2015.
All rules were read there at offset 0 (velocity at the current sample).

Every rule is written as  y[n] - y[n-K] = sum_k b_k x[n-k]  (T = 1), with group delay tau.
Complex relative error at reading offset r:  eps(w) = j w e^{j w r} H(e^{jw}) - 1.

Parts
 1. worst-case |eps| on [0, pi/2], read at offset 0 and at tau (Fig. 1(d));
 2. Example 1 of the assessment (sine acceleration): the rectangular rule attenuates
    read at offset 0 but amplifies read at tau = 1/2, as its magnitude ratio says;
 3. Example 3 (Bogdanoff's accelerogram, random phases): peak velocities are blind to the
    3-sample delay of the Schuessler-Ibler integrator, the RMS waveform error is not;
    RMS error from the amplitude spectrum by Eq. (10).

Run:   python seismic_rules.py           (numbers only)
       python seismic_rules.py --plot    (also seismic_rules.png, needs matplotlib)
Requirements: numpy, scipy (matplotlib optional).
"""
import sys
from math import ceil
import numpy as np
from scipy.signal import lfilter

SI = np.array([-0.01330, 0.07760, -0.28279, 1.71822, 1.71822, -0.28279, 0.07760, -0.01330]) / 3
RULES = {  # name: (b, K, tau)
    "Trapezoidal":      (np.array([0.5, 0.5]), 1, 0.0),
    "Simpson":          (np.array([1, 4, 1]) / 3, 2, 0.0),
    "Simpson 3/8":      (np.array([1, 3, 3, 1]) * 3 / 8, 3, 0.0),
    "Newton-Cotes 4":   (np.array([7, 32, 12, 32, 7]) * 2 / 45, 4, 0.0),
    "Rectangle":        (np.array([0.0, 1.0]), 1, 0.5),   # y[n] = y[n-1] + x[n-1]: tau = +1/2
    "Tick":             (np.array([0.3584, 1.2832, 0.3584]), 2, 0.0),
    "Schuessler-Ibler": (SI, 1, 3.0),                     # Schuessler & Ibler (1974), tau = 3
}


def H(b, K, w):
    w = np.atleast_1d(np.asarray(w, float))
    return (np.exp(-1j * np.outer(w, np.arange(len(b)))) @ b) / (1 - np.exp(-1j * K * w))


def eps(b, K, r, w):
    w = np.atleast_1d(np.asarray(w, float))
    return 1j * w * np.exp(1j * w * r) * H(b, K, w) - 1


def run_integrator(b, K, x, h):
    a = np.zeros(K + 1); a[0] = 1.0; a[K] = -1.0
    return h * lfilter(b, a, x)


# ------------------------------------------------ Part 1: worst-case |eps|
def part1():
    w = np.linspace(1e-5, np.pi / 2, 4000)
    print("1. Worst-case |eps(w)| on [0, pi/2]")
    print("   %-17s %12s %12s" % ("rule", "offset 0", "at tau"))
    for n, (b, K, tau) in RULES.items():
        e0, et = np.abs(eps(b, K, 0.0, w)).max(), np.abs(eps(b, K, tau, w)).max()
        note = "  (pole at w = pi/2)" if K == 4 else ""
        print("   %-17s %12.3g %12.3g%s" % (n, e0, et, note))
    b, K, tau = RULES["Schuessler-Ibler"]
    wl = np.linspace(1e-5, np.pi / 2, 4000)
    print("   Schuessler-Ibler at offset 0: max |eps| = %.2f (%.1f dB) in the lower half of the band"
          % (np.abs(eps(b, K, 0, wl)).max(), 20*np.log10(np.abs(eps(b, K, 0, wl)).max())))


# ------------------------------------------------ Part 2: Example 1 (sine)
A1, H1, T1, M1 = 3.0, 0.005, 1.5, 64
N1 = int(round(T1 / H1))


def ex1_velocity(b, K, lam, r, history):
    """running velocity v(t) - v(t0) read at offset r; history 'zero' (record starts at t = 0,
    as in the assessment) or 'tone' (sine also before t = 0, i.e. no start-up)."""
    w = lam * np.pi / H1
    n = np.arange(-M1, N1 + 1); t = n * H1
    x = A1 * np.sin(w * t) if history == "tone" else np.where(n >= 0, A1 * np.sin(w * t), 0.0)
    y = run_integrator(b, K, x, H1)
    n0 = int(ceil(r - 1e-12)); t0 = (n0 - r) * H1
    idx = np.arange(n0, N1 + 1, K); tt = (idx - r) * H1
    ok = tt <= T1 + 1e-12; idx, tt = idx[ok], tt[ok]
    v = lambda t: A1 / w * (1 - np.cos(w * t))
    return tt, y[idx + M1] - y[n0 + M1], v(tt) - v(t0), w


def part2(lam=0.3):
    print("\n2. Example 1, sine acceleration at w = %.1f pi (h = 5 ms): peak velocity error (%%)" % lam)
    b, K, tau = RULES["Rectangle"]
    _, vh, _, w = ex1_velocity(b, K, lam, 0.0, "zero")
    ts = np.linspace(0, T1, 30001)
    paper = 100 * (np.max(np.abs(vh)) / np.max(A1 / w * (1 - np.cos(w * ts))) - 1)
    _, vh, ve, _ = ex1_velocity(b, K, lam, tau, "tone")
    k = np.argmax(np.abs(ve)); at_tau = 100 * (vh[k] / ve[k] - 1)
    ratio = (lam*np.pi) * abs(H(b, K, lam*np.pi)[0])
    print("   Rectangle: magnitude ratio w|H| = %.4f (> 1)" % ratio)
    print("   read at offset 0 (as in the assessment): %+.2f%%  (attenuates)" % paper)
    print("   read at tau = 1/2 (midpoint rule):        %+.2f%%  (amplifies; eps = %+.2f%%)"
          % (at_tau, 100 * eps(b, K, tau, lam*np.pi)[0].real))


# ------------------------------------------------ Part 3: Example 3 (Bogdanoff's accelerogram)
B3, AL3, H3, T3 = 0.292, -0.333, 0.01, 20.0
W3 = np.linspace(6.0, 80.25, 22)
N3 = int(round(T3 / H3)); M3 = 16


def accel(t, ph):
    return np.where(t >= 0, B3 * t * np.exp(AL3 * t) * np.cos(np.outer(t, W3) + ph).sum(1), 0.0)


def vel(t, ph):
    c = AL3 + 1j * W3
    tt = np.maximum(t, 0)[:, None]
    I = np.exp(c * tt) * (tt / c - 1 / c**2) + 1 / c**2
    return np.where(t >= 0, B3 * np.real((np.exp(1j * ph) * I).sum(1)), 0.0)


def ex3_metrics(b, K, r, ph):
    n = np.arange(-M3, N3 + 1 + 8)
    y = run_integrator(b, K, accel(n * H3, ph), H3)
    k = np.arange(int(ceil(r - 1e-12)), N3 + 1); tt = (k - r) * H3
    vh, ve = y[k + M3], vel(tt, ph)
    peak = max(abs(100 * (vh.max() / ve.max() - 1)), abs(100 * (vh.min() / ve.min() - 1)))
    e = vh - ve; i = np.arange(len(e))
    for c in range(K):                               # constant of integration per interleaved sum
        e[i % K == c] -= e[i % K == c].mean()
    rms = 100 * np.sqrt(np.mean(e**2) / np.mean((ve - ve.mean())**2))
    x = accel(np.arange(0, N3 + 1) * H3, ph)         # Eq. (10) from the amplitude spectrum
    X = np.fft.rfft(x, 8 * len(x)); f = np.fft.rfftfreq(8 * len(x), H3)
    sel = (f > 0) & (f <= 20.0)
    V2 = (np.abs(X[sel]) / (2 * np.pi * f[sel]))**2
    ee = eps(b, K, r, 2 * np.pi * f[sel] * H3)
    pred = 100 * np.sqrt(np.sum(np.abs(ee)**2 * V2) / np.sum(V2))
    return peak, rms, pred


def part3(realizations=20, seed=2015):
    print("\n3. Example 3, Bogdanoff's accelerogram, %d random-phase realizations (medians, %%)" % realizations)
    rng = np.random.default_rng(seed)
    phases = [rng.uniform(0, 2 * np.pi, 22) for _ in range(realizations)]
    print("   %-17s %-7s %10s %12s %14s" % ("rule", "read at", "|peak err|", "RMS error", "RMS by (10)"))
    for n in ("Schuessler-Ibler", "Rectangle", "Trapezoidal", "Tick", "Simpson"):
        b, K, tau = RULES[n]
        for lab, r in (("0", 0.0), ("tau", tau)):
            if r == 0.0 and tau == 0.0 and lab == "tau":
                continue
            R = np.array([ex3_metrics(b, K, r, ph) for ph in phases])
            med = np.median(R, axis=0)
            print("   %-17s %-7s %10.3f %12.3f %14.3f" % (n, lab, *med))


def plot():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    lam = np.linspace(0.003, 0.997, 1200); w = lam * np.pi
    fig, ax = plt.subplots(figsize=(6, 4))
    for n, (b, K, tau) in RULES.items():
        if K == 4:
            continue
        l, = ax.plot(lam, 20*np.log10(np.abs(eps(b, K, tau, w))), lw=1, label=n + " (tau)")
        if tau != 0:
            ax.plot(lam, 20*np.log10(np.abs(eps(b, K, 0.0, w))), "--", color=l.get_color(), lw=1,
                    label=n + " (offset 0)")
    ax.set_xlim(0, 1); ax.set_ylim(-100, 20); ax.set_xlabel("w/pi"); ax.set_ylabel("20 log10 |eps(w)|")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig("seismic_rules.png", dpi=120)
    print("saved seismic_rules.png")


if __name__ == "__main__":
    part1(); part2(); part3()
    if "--plot" in sys.argv:
        plot()
