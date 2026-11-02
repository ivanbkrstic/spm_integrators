"""Fig. 1: complex relative error |eps(w)| of the integrators of both evaluations.

Companion code for the manuscript I. Krstic, "Reading the Frequency Response Right:
How to Predict the Accuracy of Digital Integrators," submitted to IEEE Signal Processing
Magazine (DSP Tips & Tricks). Equation, figure and table numbers refer to the submitted
manuscript.

Dashed: read at offset r = 0, as in the two evaluations; solid: read at the right
instant; dotted: magnitude error |w|H| - 1| of the Al-Alaoui integrator, which is what
a magnitude plot shows.  eps(w) = j w e^{j w r} H(e^{jw}) - 1  (Eq. (4)).

The nulls of |eps(w)| are exact zeros (-inf dB). To show every notch at its true
depth, the curves are evaluated on a dense grid and each local minimum is refined by
a bounded scalar minimization before plotting.

Run:   python figure1.py      (writes fig1.pdf / fig1.png)
Requirements: numpy, matplotlib.

Repository: https://github.com/ivanbkrstic/spm_integrators
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import minimize_scalar

SI = np.array([-0.01330, 0.07760, -0.28279, 1.71822, 1.71822, -0.28279, 0.07760, -0.01330]) / 3
INT = {  # name: (b, K)
    "rectangular": (np.array([1.0]), 1),
    "trapezoidal": (np.array([0.5, 0.5]), 1),
    "Simpson 1/3": (np.array([1, 4, 1]) / 3, 2),
    "Tick": (np.array([0.3584, 1.2832, 0.3584]), 2),
    "Al-Alaoui": (np.array([1 / 8, 7 / 8]), 1),
    "Schuessler-Ibler": (SI, 1),
}
COL = {"rectangular": "#8c564b", "trapezoidal": "#2ca02c", "Simpson 1/3": "#17becf", "Tick": "#e377c2",
       "Al-Alaoui": "#d62728", "Schuessler-Ibler": "#9467bd"}


def H(b, K, w):
    return (np.exp(-1j * np.outer(w, np.arange(len(b)))) @ b) / (1 - np.exp(-1j * K * w))


def eps(name, r, w):
    b, K = INT[name]
    return 1j * w * np.exp(1j * w * r) * H(b, K, w) - 1


def err_fun(name, r):
    """|eps(w)| (r given) or the magnitude-only error ||w H| - 1| (r is None) as a function of w"""
    b, K = INT[name]
    if r is None:
        return lambda w: np.abs(np.abs(H(b, K, np.atleast_1d(w))) * np.atleast_1d(w) - 1)
    return lambda w: np.abs(eps(name, r, np.atleast_1d(w)))


def curve(f, n=20001, w0=1e-4, w1=np.pi):
    """dense grid plus the refined local minima (notches) of f"""
    w = np.linspace(w0, w1, n); e = f(w)
    i = np.where((e[1:-1] < e[:-2]) & (e[1:-1] <= e[2:]))[0] + 1
    wn = [minimize_scalar(lambda v: f(v)[0], bounds=(w[k - 1], w[k + 1]), method="bounded",
                          options={"xatol": 1e-13}).x for k in i]
    w = np.sort(np.concatenate([w, wn]))
    return w, f(w)


def main():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                         "axes.grid": True, "grid.alpha": .3, "savefig.bbox": "tight"})
    db = lambda e: 20 * np.log10(np.abs(e) + 1e-14)
    curves = [  # name, r, style, width, label
        ("rectangular", 0.0, "--", 1.2, r"rectangular, $r=0$"),
        ("rectangular", -0.5, "-", 1.6, r"rectangular, $r=\tau=-\frac{1}{2}$"),
        ("trapezoidal", 0.0, "-", 1.6, "trapezoidal"),
        ("Simpson 1/3", 0.0, "-", 1.6, "Simpson 1/3"),
        ("Tick", 0.0, "-", 1.6, "Tick"),
        ("Al-Alaoui", 0.0, "--", 1.2, r"Al-Alaoui, $r=0$"),
        ("Al-Alaoui", None, ":", 1.2, "Al-Alaoui, magnitude only"),
        ("Al-Alaoui", 0.405, "-", 1.6, r"Al-Alaoui, $r=0.405$"),
        ("Schuessler-Ibler", 0.0, "--", 1.2, r"Schüssler–Ibler, $r=0$"),
        ("Schuessler-Ibler", 3.0, "-", 1.6, r"Schüssler–Ibler, $r=\tau=3$"),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for name, r, ls, lw, label in curves:
        w, e = curve(err_fun(name, r))
        ax.plot(w / np.pi, db(e), ls, color=COL[name], lw=lw, label=label)
    g = FuncFormatter(lambda v, p: f"{v:g}")
    ax.set_xlim(0, 1); ax.set_ylim(-100, 20)
    ax.set_xlabel(r"$\omega/\pi$"); ax.set_ylabel(r"$20\log_{10}|\varepsilon(\omega)|$ (dB)")
    ax.xaxis.set_major_formatter(g); ax.yaxis.set_major_formatter(g)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, handlelength=2.6)
    fig.savefig("fig1.pdf"); fig.savefig("fig1.png", dpi=130)
    print("saved fig1.pdf, fig1.png")


if __name__ == "__main__":
    main()
