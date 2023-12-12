import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift
from scipy.interpolate import interp1d

import analysis as an
from optics import optics


class SPECTRUM:
    def __init__(self, filename):
        data = np.loadtxt(filename)
        self.data = data
        self.lam = data[:, 0]
        self.I = data[:, 1]
        self.t = None
        self.I_t = None

    def calculateTransformLimited(self):
        lam = self.lam
        I = self.I
        sel = lam > 740
        f = optics.f_from_lam(lam[sel] * 1e-9)
        dlam = lam[sel][1:] - lam[sel][:-1]
        df = f[:-1] - f[1:]
        dlam_df = dlam / df
        dlam_df = np.append(dlam_df, 2 * dlam_df[-1] - dlam_df[-2])
        # Need to multiply by dlam/df to get spectral intensity in frequency
        I_f = I[sel] * dlam_df
        ff = interp1d(f, I_f, bounds_error=False, fill_value=0.0)

        N = 2**17
        f_a = np.linspace(-1e16, 1e16, N)
        I_fa = ff(f_a)
        # Remove any values below 0
        sel = I_fa < 0.0
        I_fa[sel] = 0.0
        # Find the field
        E_f = np.sqrt(I_fa)
        E_fs = ifftshift(E_f)
        E_s = ifft(E_fs)
        E = fftshift(E_s)

        T = N / (4 * f_a[0])
        t, dt = np.linspace(-T, T, N, retstep=True)
        I_t = optics.I_from_E(E)
        I_t = I_t / np.max(I_t)
        t = t * 1e15
        self.t = t
        self.I_t = I_t
        return t, I_t


def plotSpectrumFit(spec, name=None, xlim=[740, 860]):
    lam = spec.lam
    I = spec.I

    fwhm, height, start, end = an.fwhm(I, lam)
    fw_e2, height_e2, start_e2, end_e2 = an.fw_e2(I, lam)

    ylim = [0, 1.1 * np.max(I)]

    # Fit a Guassian to the measurement
    def gaussian(x, A, sigma, x_0):
        return A * np.exp(-((x - x_0) ** 2) / (2 * sigma**2))

    popt, pcov = curve_fit(gaussian, lam, I, p0=(ylim[1] / 1.1, 10.0, 800.0))

    fig = plt.figure(figsize=(6.5, 3), dpi=150)
    ax = fig.add_subplot()
    ax.plot(lam, I, label="Measured spectrum")
    ax.plot(lam, gaussian(lam, *popt), "grey", label="Gaussian fit")
    bh = ylim[1] / 30
    ax.plot([start, end], [height, height], "k")
    ax.plot([start, start], [height - bh, height + bh], "k")
    ax.plot([end, end], [height - bh, height + bh], "k")
    ax.plot([start_e2, end_e2], [height_e2, height_e2], "k")
    ax.plot([start_e2, start_e2], [height_e2 - bh, height_e2 + bh], "k")
    ax.plot([end_e2, end_e2], [height_e2 - bh, height_e2 + bh], "k")
    ax.set_xlim(xlim)
    ax.set_ylim(0, 1.1 * np.max(I))
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Spectral Intensity (counts)")
    ax.legend(frameon=False, loc="upper left")

    # ax.text(0.01, 0.94, 'Plot $x$ limits correspond to probe compressor bandwidth', transform=ax.transAxes)
    ax.text(
        end + 1,
        height,
        "FWHM={:0.2f}nm ({:0.1f}-{:0.1f}nm)".format(fwhm, start, end),
        horizontalalignment="left",
        verticalalignment="center",
    )
    ax.text(
        end_e2 + 1,
        height_e2,
        r"$1/e^2=${:0.2f}nm ({:0.1f}-{:0.1f}nm)".format(fw_e2, start_e2, end_e2),
        horizontalalignment="left",
        verticalalignment="center",
    )
    ax.text(
        0.98,
        0.95,
        r"$\lambda_0=${:0.1f}nm, $\sigma_\lambda=${:0.2f}nm".format(popt[2], popt[1]),
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )

    plt.tight_layout()
    if name is not None:
        plt.savefig(name + ".png")
    plt.show()


def plotTransformLimitedFit(spec, name=None):
    if spec.t is None:
        spec.calculateTransformLimited()
    t = spec.t
    I_t = spec.I_t
    fwhm, height, start, end = an.fwhm(I_t, t)
    fw_e2, height_e2, start_e2, end_e2 = an.fw_e2(I_t, t)

    # Fit a Guassian to the transform limited pulse
    def gaussian(x, A, sigma, x_0):
        return A * np.exp(-((x - x_0) ** 2) / (2 * sigma**2))

    popt, pcov = curve_fit(gaussian, t, I_t, p0=(1.0, 10.0, 0.0))

    fig = plt.figure(figsize=(6.5, 3), dpi=150)
    ax = fig.add_subplot()
    ax.plot(t, I_t, label="Transform limited")
    ax.plot(t, gaussian(t, *popt), "grey", label="Gaussian fit")
    bh = 0.04
    ax.plot([start, end], [height, height], "k")
    ax.plot([start, start], [height - bh, height + bh], "k")
    ax.plot([end, end], [height - bh, height + bh], "k")
    ax.set_xlim(-100, 100)
    # ax.set_ylim(0, 3000)
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Intensity (normalized)")
    ax.legend(frameon=False, loc="upper left")

    ax.text(
        start + 1,
        height,
        "FWHM={:0.2f}fs".format(-fwhm),
        horizontalalignment="left",
        verticalalignment="center",
    )
    ax.text(
        0.98,
        0.95,
        r"$\sigma_t=${:0.2f}fs".format(popt[1]),
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )

    plt.tight_layout()
    if name is not None:
        plt.savefig(name + ".png")
    plt.show()
