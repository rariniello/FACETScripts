import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import analysis as an


def getLineouts(fit_image):
    x_line = fit_image.data[int(fit_image.center[1]), :]
    x = fit_image.x
    y_line = fit_image.data[:, int(fit_image.center[0])]
    y = fit_image.y
    return x_line, x, y_line, y


def fitLineouts(fit_image, p0_x=[60, 2000, 20, 4], p0_y=[35, 2000, 20, 4]):
    x_line, x, y_line, y = getLineouts(fit_image)
    fsg, px = an.fit_superGaussian(x, x_line, p0=p0_x)
    fsg, py = an.fit_superGaussian(y, y_line, p0=p0_y)
    return fsg, px, py


def plotLineouts(fit_image, fsg, px, py, name=None):
    x_line, x, y_line, y = getLineouts(fit_image)

    fig = plt.figure(figsize=(7, 2.2), dpi=300)
    ax = fig.add_subplot(121)
    ax.plot(x - px[0], x_line, ".", markersize=2, markeredgewidth=0, label="Lineout")
    ax.plot(x - px[0], fsg(x, *px), label="Fit")
    ax.set_xlim(-2 * px[2], 2 * px[2])
    ax.set_xlabel(r"$x$ (mm)")
    ax.set_ylabel(r"Intensity (Counts)")
    ax.legend(frameon=False, prop={"size": 7}, loc=1, framealpha=1.0)
    ax.text(
        0.02,
        0.92,
        r"$f(x)=I_0\exp\left[-2\left(\left(\frac{x}{w_0}\right)^2\right)^m\right]$",
        transform=ax.transAxes,
    )
    ax.text(0.02, 0.82, r"$w_0=${:0.2f} mm".format(px[2]), transform=ax.transAxes)
    ax.text(0.02, 0.74, r"$I_0=${:0.2f}".format(px[1]), transform=ax.transAxes)
    ax.text(0.02, 0.66, r"$m=${:0.2f}".format(px[3]), transform=ax.transAxes)

    ax = fig.add_subplot(122)
    ax.plot(y - py[0], y_line, ".", markersize=2, markeredgewidth=0, label="Lineout")
    ax.plot(y - py[0], fsg(y, *py), label="Fit")
    ax.set_xlim(-2 * py[2], 2 * py[2])
    ax.set_xlabel(r"$y$ (mm)")
    ax.set_ylabel(r"Intensity (Counts)")
    ax.legend(frameon=False, prop={"size": 7}, loc=1, framealpha=1.0)
    ax.text(
        0.02,
        0.92,
        r"$f(x)=I_0\exp\left[-2\left(\left(\frac{x}{w_0}\right)^2\right)^m\right]$",
        transform=ax.transAxes,
    )
    ax.text(0.02, 0.82, r"$w_0=${:0.2f} mm".format(py[2]), transform=ax.transAxes)
    ax.text(0.02, 0.74, r"$I_0=${:0.2f}".format(py[1]), transform=ax.transAxes)
    ax.text(0.02, 0.66, r"$m=${:0.2f}".format(py[3]), transform=ax.transAxes)
    plt.tight_layout()
    if name is not None:
        plt.savefig(name)
    plt.show()


def fitTotal(fit_image, popt, name=None):
    # Fit a single super-Gaussian to the entire image

    def f2_2(x, y, x_0, y_0, I_0, w_0, m):
        r = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2)
        return I_0 * np.exp(-2 * ((r / w_0) ** 2) ** m)

    X, Y = np.meshgrid(fit_image.x, fit_image.y)
    Z = fit_image.data

    ext = fit_image.get_ext(cal=True)
    x_0 = popt[0]
    y_0 = popt[1]
    ext2 = np.copy(ext)
    ext2[0] -= x_0
    ext2[1] -= x_0
    ext2[2] -= y_0
    ext2[3] -= y_0

    fig = plt.figure(figsize=(6.5, 3), dpi=300)
    ax = fig.add_subplot(121)
    vmin = 0
    vmax = np.amax(Z)
    cmap = "inferno"
    ax.imshow(Z, extent=ext2, vmin=vmin, vmax=vmax, cmap=cmap)
    phi = np.linspace(0, 2 * np.pi, 1000)
    ax.plot(popt[3] * np.cos(phi), popt[3] * np.sin(phi), "--", label=r"$w_0$")
    ax.set_xlim(-1.5 * popt[3], 1.5 * popt[3])
    ax.set_ylim(1.5 * popt[3], -1.5 * popt[3])
    ax.set_xlabel("$x$ (mm)")
    ax.set_ylabel("$y$ (mm)")
    ax.text(0.02, 0.94, "Raw Image", transform=ax.transAxes, color="w")
    ax.legend(frameon=False, prop={"size": 7}, loc=1, framealpha=1.0, labelcolor="w")
    ax.set_facecolor("k")

    ax2 = fig.add_subplot(122)
    ax2.imshow(f2_2(X, Y, *popt), extent=ext2, vmin=vmin, vmax=vmax, cmap=cmap)
    ax2.plot(popt[3] * np.cos(phi), popt[3] * np.sin(phi), "--", label=r"$w_0$")
    ax2.set_xlim(-1.5 * popt[3], 1.5 * popt[3])
    ax2.set_ylim(1.5 * popt[3], -1.5 * popt[3])
    ax2.set_xlabel("$x$ (mm)")
    ax2.set_ylabel("$y$ (mm)")
    ax2.text(0.02, 0.94, "Super-Gaussian Fit", transform=ax2.transAxes, color="w")
    ax2.text(
        0.02,
        0.88,
        r"$f(r)=I_0\exp\left[-2\left(\left(\frac{r}{w_0}\right)^2\right)^m\right]$",
        transform=ax2.transAxes,
        color="w",
    )
    ax2.text(
        0.02,
        0.82,
        r"$w_0=${:0.2f} mm".format(popt[3]),
        transform=ax2.transAxes,
        color="w",
    )
    ax2.text(
        0.02, 0.76, r"$I_0=${:0.2f}".format(popt[2]), transform=ax2.transAxes, color="w"
    )
    ax2.text(
        0.02, 0.70, r"$m=${:0.2f}".format(popt[4]), transform=ax2.transAxes, color="w"
    )
    ax2.legend(frameon=False, prop={"size": 7}, loc=1, framealpha=1.0, labelcolor="w")
    ax2.set_facecolor("k")
    print("Pixel sum image: {:0.2e}".format(np.sum(Z)))
    print("Pixel sum fit  : {:0.2e}".format(np.sum(f2_2(X, Y, *popt))))
    plt.tight_layout()
    if name is not None:
        plt.savefig(name)
    plt.show()
