import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib as mpl

import simulation
import fourierProp as fp
from optics import optics


def fourierTransformField(E: np.ndarray) -> np.ndarray:
    """Fourier transforms the field.

    Args:
        E: Field in real space.

    Returns:
        Numpy arry containing the shifted Fourier transform of the field.
    """
    return np.fft.fftshift(np.fft.fft2(E))


def plotFieldAtInd(path, ind):
    attrs, data = fp.load.loadGridAtPlane(path, ind=ind)
    x = data["x"]["x"]
    y = data["y"]["y"]
    fx = data["x"]["fx"]
    fy = data["y"]["fy"]
    field = fp.load.loadFieldAtPlane(path, ind=ind)
    params = fp.load.loadSimulation(path)
    if params["cylSymmetry"]:
        raise NotImplementedError("Resampling to 2D not implemented.")
    plotField(x, y, fx, fy, field)


def plotField(x, y, fx, fy, field):
    Nx = len(x)
    Ny = len(y)
    ext = simulation.extentFromGrid(x, y) * 1e3
    extf = simulation.extentFromGrid(fx, fy) * 1e-3
    phi = np.angle(field)
    I = optics.I_from_E(field)
    e = fourierTransformField(field)

    fig = plt.figure(figsize=(7.5, 4), dpi=300)
    ax = fig.add_subplot(231)
    im = ax.imshow(np.rot90(I * 1e-4), extent=ext, cmap="inferno")
    ax.set_xlabel(r"$x$ (mm)")
    ax.set_ylabel(r"$y$ (mm)")
    cb = fig.colorbar(im)
    cb.set_label(r"Intensity ($\mathrm{W/cm^2}$)")

    ax = fig.add_subplot(232)
    im = ax.imshow(np.rot90(phi), extent=ext, cmap="RdYlBu")
    ax.set_xlabel(r"$x$ (mm)")
    ax.set_ylabel(r"$y$ (mm)")
    cb = fig.colorbar(im)
    cb.set_label(r"Phase (rad)")

    ax = fig.add_subplot(233)
    im = ax.imshow(np.rot90(abs(e) ** 2), extent=extf)
    ax.set_xlabel(r"$f_x$ ($\mathrm{1/mm}$)")
    ax.set_ylabel(r"$f_y$ ($\mathrm{1/mm}$)")
    cb = fig.colorbar(im)
    cb.set_label(r"Intensity (arb. units)")

    ax = fig.add_subplot(234)
    ax.plot(x * 1e3, I[:, int(Ny / 2)] * 1e-4, label=r"$x$ lineout")
    ax.plot(y * 1e3, I[int(Nx / 2), :] * 1e-4, label=r"$y$ lineout")
    ax.set_xlabel(r"$x$ or $y$ (mm)")
    ax.set_ylabel(r"Intensity ($\mathrm{W/cm^2}$)")
    ax.legend(loc=1, frameon=False, fontsize=6)

    ax = fig.add_subplot(235)
    ax.plot(x * 1e3, np.unwrap(phi[:, int(Ny / 2)]), label=r"$x$ lineout")
    ax.plot(y * 1e3, np.unwrap(phi[int(Nx / 2), :]), label=r"$y$ lineout")
    ax.set_xlabel(r"$x$ or $y$ (mm)")
    ax.set_ylabel(r"Phase (rad)")
    ax.legend(loc=1, frameon=False, fontsize=6)

    ax = fig.add_subplot(236)
    ax.plot(fx * 1e-3, abs(e[:, int(Ny / 2)]) ** 2, label=r"$x$ lineout")
    ax.plot(fy * 1e-3, abs(e[int(Nx / 2), :]) ** 2, label=r"$y$ lineout")
    ax.set_xlabel(r"$f_x$ or $f_y$ ($\mathrm{1/mm}$)")
    ax.set_ylabel(r"Intensity (arb. units)")
    ax.legend(loc=1, frameon=False, fontsize=6)

    plt.tight_layout()
    plt.show()
