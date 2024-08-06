import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from scipy.constants import physical_constants

c = physical_constants["speed of light in vacuum"][0]

import pulseProp.reconstruct as rc
import pulseProp as pp
from plot import cmap_RdWBu, cmap_RdW, roar
from optics import optics


def plotInitialPulse(path):
    pulseAttrs, pulseData = pp.load.loadPulse(path)
    t = pulseData["t"]
    I_t = abs(pulseData["E_t"]) ** 2

    E_tr = rc.reconstructInitialPulse(path)
    I_tr = abs(E_tr) ** 2

    t = pulseData["t"]
    I_t = abs(pulseData["E_t"]) ** 2

    xlim = [t[0] * 1e15, t[-1] * 1e15]
    fig = plt.figure(figsize=(7.5, 2.5), dpi=300)
    ax = fig.add_subplot(121)
    ax.plot(t * 1e15, I_t, label="Original")
    ax.plot(t * 1e15, I_tr, "--", label="Propagated")
    ax.set_xlabel(r"$t$ (fs)")
    ax.set_ylabel("Intensity (normalized)")
    ax.set_xlim(xlim)
    ax.legend()

    ax2 = fig.add_subplot(122)
    ax2.semilogy(t * 1e15, I_t, label="Original")
    ax2.plot(t * 1e15, I_tr, "--", label="Propagated")
    ax2.set_ylim(1e-10, 2)
    ax2.set_xlabel(r"$t$ (fs)")
    ax2.set_ylabel("Intensity (normalized)")
    ax2.set_xlim(xlim)
    ax2.legend()

    plt.show()


def plotField(E, ext, xlim=None, ylim=None):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    im = ax.imshow(np.flipud(E.real), cmap=cmap_RdWBu, aspect="auto", extent=ext)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(r"$t$ (fs)")
    ax.set_ylabel(r"$x$ (mm)")
    cb = plt.colorbar(im)
    cb.set_label("$E$ (V/m)")
    plt.show()


def plotIntensity(E, ext, logScale=False, xlim=None, ylim=None, clim=None):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    if logScale:
        im = ax.imshow(
            np.flipud(optics.I_from_E(E)) / 1e4,
            cmap=cmap_RdW,
            aspect="auto",
            extent=ext,
            norm=mpl.colors.LogNorm(),
        )
    else:
        im = ax.imshow(
            np.flipud(optics.I_from_E(E)) / 1e4,
            cmap=cmap_RdW,
            aspect="auto",
            extent=ext,
        )
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(r"$t$ (fs)")
    ax.set_ylabel(r"$x$ (mm)")
    cb = plt.colorbar(im)
    cb.set_label("$I$ ($\mathrm{W/cm^2}$)")
    if clim is not None:
        im.set_clim(clim)
    plt.show()


def plotFourierSpace(U, extf, xlim=None, ylim=None, xWavelength=False):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    im = ax.imshow(np.flipud(abs(U) ** 2), cmap=cmap_RdW, aspect="auto", extent=extf)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    cb = plt.colorbar(im)
    cb.set_label("Intensity (arb. units)")
    ax.set_xlabel(r"$f$ (Hz)")
    ax.set_ylabel(r"$f_x$ (1/mm)")
    if xWavelength:
        tick_locs = ax.get_xticks()
        tick_lbls = ["{:0.1f}".format(c / i * 1e9) for i in tick_locs]
        plt.xticks(tick_locs, tick_lbls)
        ax.set_xlabel(r"$\lambda$ (nm)")
    plt.show()
