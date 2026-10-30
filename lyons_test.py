"""Lyons' integrator test re-read with the complex relative error.

Companion code for I. Krstic, "Reading the Frequency Response Right: How to Predict
the Accuracy of Digital Integrators" (IEEE SPM, DSP Tips & Tricks).

Test: R. Lyons, "The risk in using frequency domain curves to evaluate digital
integrator performance," DSPRelated.com, 2019 - seven test signals, N = 24 intervals.

Integrator  H(z) = B(z)/(1 - z^-K),   y[n] - y[n-K] = sum_k b_k x[n-k].
The output y[n] is read as the integral up to (n - r)T  (reading offset r), and the
definite integral over [a, b] is  I_hat = T (y[n0+N] - y[n0]),  a = (n0 - r)T.
Real time: the samples needed beyond each end of [a, b] are true samples of f.

Complex relative error (Eq. (4)):  eps(w) = j w e^{j w r} H(e^{jw}) - 1.

Run:   python lyons_test.py          (numbers only)
       python lyons_test.py --plot   (also lyons_test.png, needs matplotlib)
Requirements: numpy, scipy (matplotlib optional).
"""
import sys
import numpy as np
from scipy.integrate import quad


# ----------------------------------------------------------------- integrators
def mf(L, K):
    """Maximally flat linear-phase integrator MF(L,K), closed form (2):
    b_k = int_tau^{tau+K} prod_{i != k} (u - i)/(k - i) du,  tau = (L-1-K)/2."""
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


# name: (b, K, reading offset r used in the test)
INTEGRATORS = {
    "Rectangular": (np.array([1.0]), 1, 0.0),          # group delay tau = -1/2, read at 0 (as Lyons)
    "Al-Alaoui":   (np.array([1/8, 7/8]), 1, 0.0),     # not linear phase, read at 0 (as Lyons)
    "Trapezoidal": (np.array([0.5, 0.5]), 1, 0.0),     # tau = 0
    "Tick":        (np.array([0.3584, 1.2832, 0.3584]), 2, 0.0),
    "Simpson 1/3": (np.array([1/3, 4/3, 1/3]), 2, 0.0),
    "MF(4,1)":     (mf(4, 1), 1, 1.0),                 # read at tau = 1
    "MF(5,2)":     (mf(5, 2), 2, 1.0),                 # read at tau = 1
}


def H(b, K, w):
    w = np.atleast_1d(np.asarray(w, float))
    B = np.exp(-1j * np.outer(w, np.arange(len(b)))) @ b
    return B / (1 - np.exp(-1j * K * w))


def eps(b, K, r, w):
    """complex relative error (4) at reading offset r"""
    w = np.atleast_1d(np.asarray(w, float))
    return 1j * w * np.exp(1j * w * r) * H(b, K, w) - 1


def alpha1(b, K, r):
    """first-order Taylor coefficient of eps:  eps(w) = alpha1 (j w) + ..."""
    return r + K / 2 - np.dot(np.arange(len(b)), b) / K


# ----------------------------------------------------------------- test signals
SIGNALS = [  # (f, a, b, label)
    (lambda t: 1 / t,                                     1.0, 2.0,          "1/t"),
    (lambda t: t * np.exp(-t**2),                         0.0, np.pi,        "t exp(-t^2)"),
    (lambda t: np.sin(t),                                 0.0, 13*np.pi/3,   "sin(t)"),
    (lambda t: np.sin(t) + 0.25,                          0.0, 4.0,          "sin(t)+0.25"),
    (lambda t: t*np.exp(-t**2) + 0.2*np.sin(10*t/3),      0.0, np.pi,        "t exp(-t^2)+0.2 sin(10t/3)"),
    (lambda t: np.exp(-t/10) * np.cos(t),                 0.0, 7*np.pi/3,    "exp(-t/10) cos(t)"),
    (lambda t: np.abs(np.sin(t)),                         0.0, 2.22*np.pi,   "|sin(t)|"),
]
N = 24


def true_integral(f, a, b):
    pts = [np.pi, 2*np.pi] if b > 2*np.pi and a == 0 else None
    import warnings; warnings.filterwarnings("ignore")
    return quad(f, a, b, points=pts, limit=400, epsabs=1e-15, epsrel=1e-15)[0]


def estimate(f, a, b_end, bk, K, r, N=N):
    """I_hat = T (y[N] - y[0]) with n0 = 0, so a = -rT and x[m] = f(a + (m + r) T).
    y[N] - y[0] = sum over n = K, 2K, ..., N of sum_k b_k x[n - k]  (N must be a multiple of K)."""
    assert N % K == 0
    T = (b_end - a) / N
    x = lambda m: f(a + (m + r) * T)
    s = 0.0
    for n in range(K, N + 1, K):
        s += sum(bk[k] * x(n - k) for k in range(len(bk)))
    return T * s


def pct(est, I):
    return 100 * abs(est - I) / abs(I)


# ----------------------------------------------------------------- main
def main():
    np.set_printoptions(precision=4)
    I = [true_integral(f, a, b) for f, a, b, _ in SIGNALS]

    print("Relative errors (%) of the definite integrals, N = 24, real time")
    print("%-26s" % "signal" + "".join("%13s" % n for n in INTEGRATORS))
    E = {}
    for s, (f, a, b, lab) in enumerate(SIGNALS):
        row = []
        for n, (bk, K, r) in INTEGRATORS.items():
            e = pct(estimate(f, a, b, bk, K, r), I[s])
            E.setdefault(n, []).append(e); row.append(e)
        print("%-26s" % lab + "".join("%13.3g" % e for e in row))

    bA, KA, _ = INTEGRATORS["Al-Alaoui"]
    bR, KR, _ = INTEGRATORS["Rectangular"]
    aa38 = [pct(estimate(f, a, b, bA, KA, 3/8), I[s]) for s, (f, a, b, _) in enumerate(SIGNALS)]
    print("%-26s" % "Al-Alaoui read at r = 3/8", " ".join("%.3g" % e for e in aa38))

    print("\nLeading terms of eps(w) (Table 1):")
    for n, (bk, K, r) in INTEGRATORS.items():
        print("  %-13s r = %-4g alpha1 = %+.4f" % (n, r, alpha1(bk, K, r)))
    print("  %-13s r = %-4g alpha1 = %+.4f" % ("Al-Alaoui", 3/8, alpha1(bA, KA, 3/8)))

    # shift correction (9): I_corr = I_hat - alpha1 T (x[N] - x[0])  (r = 0)
    print("\nShift correction (9): both integrators turn into the trapezoidal rule")
    f, a, b, _ = SIGNALS[3]; T = (b - a) / N
    for n, (bk, K, r) in (("Rectangular", INTEGRATORS["Rectangular"]), ("Al-Alaoui", INTEGRATORS["Al-Alaoui"])):
        Ih = estimate(f, a, b, bk, K, r)
        Ic = Ih - alpha1(bk, K, r) * T * (f(b) - f(a))
        print("  sin(t)+0.25, %-12s %.3f%%  ->  corrected %.3f%%" % (n, pct(Ih, I[3]), pct(Ic, I[3])))
    bT, KT, rT = INTEGRATORS["Trapezoidal"]
    print("  sin(t)+0.25, trapezoidal   %.3f%%" % pct(estimate(f, a, b, bT, KT, rT), I[3]))
    print("  sin(t)+0.25, Al-Alaoui at r = 3/8: %.3f%%, Tick: %.3f%%"
          % (aa38[3], E["Tick"][3]))

    # real sinusoid: I_hat - I = Re{eps(w0) e^{j theta} J(Omega0)}
    f, a, b, _ = SIGNALS[2]; T = (b - a) / N; w0 = T            # Omega0 = 1
    theta = -np.pi / 2                                           # sin(t) = cos(t - pi/2)
    J = (np.exp(1j * b) - np.exp(1j * a)) / 1j
    e0 = eps(bA, KA, 0.0, w0)[0]
    Ipred = np.real(np.exp(1j*theta) * J) + np.real(e0 * np.exp(1j*theta) * J)
    print("\nsin(t) on [0, 13pi/3], Al-Alaoui at r = 0:")
    print("  measured error %.1f%%, from Re{eps e^{j theta} J}: %.1f%%, |eps(w0)| = %.0f%% at w0 = 13pi/72"
          % (E["Al-Alaoui"][2], pct(Ipred, I[2]), 100 * abs(e0)))

    # best reading instant of the Al-Alaoui integrator on [0, pi/2]
    w = np.linspace(1e-5, np.pi / 2, 4000)
    rs = np.linspace(0, 1, 2001)
    m = [np.abs(eps(bA, KA, r, w)).max() for r in rs]
    ropt = rs[int(np.argmin(m))]
    print("\nAl-Alaoui, worst-case |eps| on [0, pi/2]:")
    print("  r = 0: %.0f%%   r = 3/8: %.1f%%   r_opt = %.3f: %.1f%%   (trapezoidal: %.0f%%)"
          % (100*np.abs(eps(bA, KA, 0, w)).max(), 100*np.abs(eps(bA, KA, 3/8, w)).max(),
             ropt, 100*min(m), 100*np.abs(eps(bT, KT, 0, w)).max()))

    # magnitude error never exceeds |eps|
    wf = np.linspace(1e-4, np.pi, 3000)
    for n, (bk, K, r) in INTEGRATORS.items():
        mag = np.abs(np.abs(H(bk, K, wf)) * wf - 1); ce = np.abs(eps(bk, K, r, wf))
        assert np.all(mag <= ce + 1e-12)
    print("\nCheck: | w|H| - 1 | <= |eps(w)| for all integrators: OK")

    if "--plot" in sys.argv:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
        for n, (bk, K, r) in INTEGRATORS.items():
            if n.startswith("MF"):
                continue
            ax[0].plot(wf/np.pi, 20*np.log10(np.abs(np.abs(H(bk, K, wf))*wf - 1) + 1e-12), label=n)
            ax[1].plot(wf/np.pi, 20*np.log10(np.abs(eps(bk, K, r, wf)) + 1e-12), label=n)
        for i, n in enumerate(INTEGRATORS):
            ax[2].plot(np.arange(1, 8), 20*np.log10(np.array(E[n])/100), "o", ms=4, label=n)
        ax[2].plot(np.arange(1, 8), 20*np.log10(np.array(aa38)/100), "s", ms=3, label="Al-Alaoui, r = 3/8")
        ax[0].set_title("(a) magnitude error (dB)"); ax[1].set_title("(b) |eps(w)| (dB)")
        ax[2].set_title("(c) errors of the integrals (dB)")
        for x in ax[:2]:
            x.set_xlabel("w/pi"); x.set_xlim(0, 1); x.set_ylim(-80, 20)
        ax[2].set_xlabel("test signal #"); ax[1].legend(fontsize=7, loc="lower right"); ax[2].legend(fontsize=6)
        fig.tight_layout(); fig.savefig("lyons_test.png", dpi=120)
        print("saved lyons_test.png")


if __name__ == "__main__":
    main()
