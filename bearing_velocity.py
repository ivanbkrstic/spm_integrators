"""Real-time velocity from measured bearing vibration (CWRU Bearing Data Center).

Companion code for the manuscript I. Krstic, "Reading the Frequency Response Right:
How to Predict the Accuracy of Digital Integrators," submitted to IEEE Signal Processing
Magazine (DSP Tips & Tricks). Equation, figure and table numbers refer to the submitted
manuscript.

Reproduces Fig. 2 and the agreement of Eq. (10) with the measured errors.

Data: drive-end accelerometer records 97 (normal), 109 (inner-race fault) and 135
(outer-race fault), 48 kHz, 1796 rpm, from https://engineering.case.edu/bearingdatacenter
Download 97.mat, 109.mat and 135.mat into one folder.

Protocol
 1. band-limit the 48 kHz acceleration to 10 Hz - 1 kHz in the frequency domain;
 2. reference velocity: each DFT coefficient divided by j*Omega, exact at any instant
    (also between samples, for fractional reading instants);
 3. downsample by 15 (fs = 3.2 kHz, no aliasing) and run the integrators sample by sample,
    each read at its reading instant r (latency);
 4. relative RMS error of the running velocity, constant of integration removed
    (for K = 2, the error is evaluated on the samples of one parity, i.e., within one of
    the two interleaved running sums, which removes the constant of integration of that
    sum; this is equivalent to removing it separately at even and odd samples);
 5. the same error from the amplitude spectrum by Eq. (10):
        rho = sqrt( sum |eps(w_k)|^2 |V_k|^2 / sum |V_k|^2 ),  |V_k| = |A_k| / Omega_k.

Run:   python bearing_velocity.py <folder with 97.mat, 109.mat, 135.mat>
       (writes fig2.pdf / fig2.png, needs matplotlib)
Requirements: numpy, scipy, matplotlib.

Repository: https://github.com/ivanbkrstic/spm_integrators
"""
import sys
from math import ceil
import numpy as np
import scipy.io as sio
from scipy.optimize import linprog
from scipy.signal import lfilter

G, FS0, D, FMIN, FMAX = 9.80665, 48000.0, 15, 10.0, 1000.0


# ----------------------------------------------------------------- integrators
def mf(L, K):
    """maximally flat linear-phase integrator MF(L,K), closed form (3)"""
    tau = (L - 1 - K) / 2
    b = []
    for k in range(L):
        p = np.poly1d([1.0])
        for i in range(L):
            if i != k:
                p *= np.poly1d([1.0, -i]) / (k - i)
        P = p.integ()
        b.append(P(tau + K) - P(tau))
    return np.array(b)


def chebyshev(L, K, wp, ngrid=4000):
    """linear-phase integrator minimizing max |eps(w)|/w on [0, wp], with eps(0) = 0 (LP)"""
    te = (L - 1) / 2; m = te - np.arange(int(np.floor(te)) + 1); M = len(m)
    w = np.linspace(1e-4, wp, ngrid)
    A = 2 * np.cos(np.outer(w, m)) / (K * np.sinc(K * w / (2 * np.pi)))[:, None] / w[:, None]
    Aub = np.vstack([np.c_[A, -np.ones(ngrid)], np.c_[-A, -np.ones(ngrid)]])
    res = linprog(np.r_[np.zeros(M), 1.0], A_ub=Aub, b_ub=np.r_[1 / w, -1 / w],
                  A_eq=np.r_[np.ones(M), 0.0][None, :], b_eq=[K / 2], bounds=[(None, None)] * (M + 1))
    g = res.x[:M].copy()
    if L % 2:
        g[-1] *= 2
    return np.concatenate([g, g[::-1][(L % 2):]])


def H(b, K, w):
    w = np.atleast_1d(np.asarray(w, float))
    return (np.exp(-1j * np.outer(w, np.arange(len(b)))) @ b) / (1 - np.exp(-1j * K * w))


def eps(b, K, r, w):
    w = np.atleast_1d(np.asarray(w, float))
    return 1j * w * np.exp(1j * w * r) * H(b, K, w) - 1


def best_offset(b, K, wp, ngrid=3000):
    """reading offset r in [0, 1] minimizing max |eps(w)| on [0, wp]"""
    w = np.linspace(1e-4, wp, ngrid); rs = np.linspace(0, 1, 1001)
    return float(rs[int(np.argmin([np.abs(eps(b, K, r, w)).max() for r in rs]))])


def integrator_list(wp):
    AA = np.array([1/8, 7/8])
    return [  # (name, b, K, reading offset r)
        ("Rectangular",            np.array([1.0]), 1, 0.0),
        ("Rectangular, r = -1/2",  np.array([1.0]), 1, -0.5),
        ("Al-Alaoui",              AA, 1, 0.0),
        ("Al-Alaoui, r = %.2f" % best_offset(AA, 1, wp), AA, 1, best_offset(AA, 1, wp)),
        ("Trapezoidal",            np.array([0.5, 0.5]), 1, 0.0),
        ("Tick",                   np.array([0.3584, 1.2832, 0.3584]), 2, 0.0),
        ("Simpson 1/3",            np.array([1, 4, 1]) / 3, 2, 0.0),
        ("MF(4,1)",                mf(4, 1), 1, 1.0),
        ("MF(5,2)",                mf(5, 2), 2, 1.0),
        ("MF(3,1)",                mf(3, 1), 1, 0.5),
        ("MF(5,1)",                mf(5, 1), 1, 1.5),
        ("Chebyshev(5,1)",         chebyshev(5, 1, wp), 1, 1.5),
        ("MF(5,1), delay ignored", mf(5, 1), 1, 0.0),
    ]


# ----------------------------------------------------------------- experiment
def experiment(acc48):
    fs = FS0 / D; T = 1 / fs; wp = 2 * np.pi * FMAX / fs
    n0 = len(acc48)
    Xf = np.fft.rfft(acc48 - acc48.mean()); ff = np.fft.rfftfreq(n0, 1 / FS0)
    Xf[(ff < FMIN) | (ff > FMAX)] = 0.0
    xb = np.fft.irfft(Xf, n0)                                   # band-limited acceleration
    Vf = np.zeros(len(Xf), complex); nz = ff > 0
    Vf[nz] = Xf[nz] / (2j * np.pi * ff[nz])                     # velocity spectrum
    ref = lambda c: np.fft.irfft(Vf * np.exp(2j * np.pi * ff * c / FS0), n0)   # v(t + c/FS0)
    xd = xb[::D]
    band = (ff >= FMIN) & (ff <= FMAX) & (np.abs(Vf) > 0)       # line spectrum for Eq. (10)
    V2, wk = np.abs(Vf[band])**2, 2 * np.pi * ff[band] * T
    out = {}
    m0 = 200                                                    # skip the start-up
    for name, b, K, r in integrator_list(wp):
        a = np.zeros(K + 1); a[0] = 1.0; a[K] = -1.0
        y = T * lfilter(b, a, xd)
        n1 = int(ceil(r - 1e-12)); s0 = n1 - r                  # y[n1 + k] refers to time (k + s0) T
        ks = np.arange(m0, len(xd) - n1 - 10, K)                # K = 2: one parity only
        est = y[n1 + ks] - y[n1 + ks[0]]
        tru = ref(s0 * D)[ks * D]; tru = tru - tru[0]
        e = est - tru; e -= e.mean(); tt = tru - tru.mean()
        rho = np.sqrt(np.sum(np.abs(eps(b, K, r, wk))**2 * V2) / np.sum(V2))
        out[name] = (100 * np.sqrt(np.mean(e**2) / np.mean(tt**2)), 100 * rho, r, b, K)
    spec = (ff[band], np.abs(Vf[band]))
    return out, spec, fs


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    folder = args[0] if args else "."
    results = {}
    worst = 0.0
    for rec in ("97", "109", "135"):
        m = sio.loadmat(f"{folder}/{rec}.mat")
        key = [k for k in m if k.endswith("_DE_time")][0]
        out, spec, fs = experiment(m[key].ravel() * G)
        results[rec] = (out, spec, fs)
        print("Record %s, fs = %.0f Hz, band %.0f Hz - %.0f Hz" % (rec, fs, FMIN, FMAX))
        print("   %-24s %12s %14s %10s" % ("integrator", "measured %", "by (10) %", "latency"))
        for name, (meas, pred, r, _, _) in out.items():
            print("   %-24s %12.4f %14.4f %10g" % (name, meas, pred, r))
            worst = max(worst, abs(pred / meas - 1))
    print("Largest deviation of (10) from the measured errors: %.2f%% of their values" % (100 * worst))
    wp = 2 * np.pi * FMAX / (FS0 / D)
    print("Chebyshev(5,1) coefficients b0, b1, b2 = %s" % ", ".join("%.6f" % v for v in chebyshev(5, 1, wp)[:3]))
    plot(results["109"][0])


def plot(out):
    """Fig. 2: relative RMS error of the running velocity, record 109"""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                         "axes.grid": True, "grid.alpha": .3, "savefig.bbox": "tight"})
    COL = {"Rectangular": "#8c564b", "Al-Alaoui": "#d62728", "Trapezoidal": "#2ca02c", "Tick": "#e377c2",
           "Simpson 1/3": "#17becf", "MF(4,1)": "#9467bd", "MF(5,2)": "#bcbd22", "MF(3,1)": "#393b79",
           "MF(5,1)": "#ff7f0e", "Chebyshev(5,1)": "#000000", "MF(5,1), delay ignored": "#7b4173"}
    names = list(out); vals = [out[n][0] for n in names]
    label = {"Rectangular": r"Rectangular, $r=0$", "Al-Alaoui": r"Al-Alaoui, $r=0$",
             "MF(5,1), delay ignored": r"MF(5,1), $r=0$ (delay ignored)"}
    def lab(n):
        if n in label: return label[n]
        if n.startswith("Rectangular, r"): return r"Rectangular, $r=-\frac{1}{2}$ (midpoint)"
        if n.startswith("Al-Alaoui, r"): return r"Al-Alaoui, $r=%s$" % n.split("= ")[1]
        return n
    fmt = lambda v: f"{v:.0f}" if v >= 9.95 else (lambda t: t[:-2] if t.endswith(".0") else t)(f"{v:#.2g}")
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    yy = np.arange(len(names))[::-1]
    bars = ax.barh(yy, vals, height=.62, color=[COL.get(n, COL.get(n.split(", r")[0])) for n in names])
    bars[-1].set_hatch("////"); bars[-1].set_edgecolor("white")
    for b_, n in zip(bars, names):
        if ", r =" in n: b_.set_alpha(.55)
    for y_, n, v in zip(yy, names, vals):
        r = out[n][2]
        ax.text(v * 1.15, y_, fmt(v) + "%" + (f" (latency {r:g})" if r and ", r =" not in n else ""),
                va="center", fontsize=7)
    ax.set_yticks(yy); ax.set_yticklabels([lab(n) for n in names], fontsize=7.5)
    ax.set_xscale("log"); ax.set_xlim(2e-2, 4e2); ax.grid(axis="y", visible=False)
    ax.set_xlabel("relative RMS error of the running velocity (%)")
    fig.savefig("fig2.pdf"); fig.savefig("fig2.png", dpi=130)
    print("saved fig2.pdf, fig2.png")


if __name__ == "__main__":
    main()
