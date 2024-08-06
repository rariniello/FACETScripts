import sys
import argparse

import numpy as np
from scipy.signal import find_peaks, medfilt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import load
import image
import dataset
import spec

from scipy.constants import physical_constants

c = physical_constants["speed of light in vacuum"][0]
e = physical_constants["elementary charge"][0]
me = physical_constants["electron mass"][0]
eps_0 = physical_constants["vacuum electric permittivity"][0]


def get_waterfall(ds, cam, satThres, satPx, crop=None):
    N = ds.N
    bkgd = ds.getImageBackground(cam)
    img = ds.getImage(cam, 0)
    wf = np.zeros((img.height, N))
    saturated = np.full(N, False)
    for i in range(N):
        img = ds.getImage(cam, i)
        img.subtract_background(bkgd)
        if crop is not None:
            img.crop(crop)
        sel = img.data > satThres
        nSatPx = np.sum(sel)
        if nSatPx > satPx:
            saturated[i] = True
        wf[:, i] = np.sum(img.data, axis=1)
    return wf, saturated, img


def find_beam_center(ds, wf, img, height, size=5):
    peaks = np.zeros(ds.N, dtype="int")
    for i in range(ds.N):
        proj = wf[:, i]
        proj = medfilt(proj, size)
        peak = find_peaks(proj, height=height)
        if len(peak[0]) > 1:
            print(i, peak[0])
        peaks[i] = peak[0][0]
    y = img.y[-1] - img.y[peaks]
    return y


# -----------------------------------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------------------------------


def plot_correlations(ds, toro3163, sumCounts, keep, saturated):
    steps = ds.common_stepIndex
    unique_steps = np.unique(steps)
    M = len(unique_steps)
    fig = plt.figure(figsize=(14, 2), dpi=150)
    gs = gridspec.GridSpec(1, M, wspace=0)
    axs = []
    for i in range(M):
        step = unique_steps[i]
        if i == 0:
            ax = fig.add_subplot(gs[0, i])
            ax.set_ylabel("TORO:LI20:2040:TMIT (nC)")
        else:
            ax = fig.add_subplot(gs[0, i], sharey=axs[0])
        # Current step not saturated and kept
        sel = (steps == step) * (~saturated) * keep
        ax.plot(
            sumCounts[sel],
            toro3163[sel] * 1e9,
            ".",
            markersize=4,
            label="Unsaturated",
            color="tab:blue",
        )
        # Current step saturated
        sel = (steps == step) * (saturated) * keep
        ax.plot(
            sumCounts[sel],
            toro3163[sel] * 1e9,
            ".",
            markersize=4,
            label="Saturated",
            color="tab:red",
        )
        # Current step not saturated and rejected
        sel = (steps == step) * (~saturated) * (~keep)
        ax.plot(
            sumCounts[sel],
            toro3163[sel] * 1e9,
            "x",
            markersize=4,
            label="Rejected",
            color="tab:blue",
        )
        # Current step saturated and rejected
        sel = (steps == step) * (saturated) * (~keep)
        ax.plot(sumCounts[sel], toro3163[sel] * 1e9, "x", markersize=4, color="tab:red")

        ax.set_xlabel(r"Sum Counts")
        ax.annotate(
            r"$E_{bend}=$" + "{:0.2f} GeV".format(ds.scan_vals[i]),
            xy=(1, 1),
            xytext=(-3, -3),
            xycoords="axes fraction",
            textcoords="offset points",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
        axs.append(ax)
    d = 1e9 * (np.max(toro3163) - np.min(toro3163))
    axs[0].set_ylim(
        np.min(toro3163) * 1e9 - 0.05 * d, np.max(toro3163) * 1e9 + 0.15 * d
    )
    ax.legend(frameon=True, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3)
    plt.setp([a.get_yticklabels() for a in axs[1:]], visible=False)
    plt.show()


def plot_calibration(
    ds, cam, y, y_avg, chargeCal, y_std, chargeCalErr, E_bend, p, d_nom, dy
):
    fig, ax = ds.plotScalarByStep(scalar=y)
    fig.set_size_inches(8, 3)
    gs = gridspec.GridSpec(1, 2)
    ax.set_subplotspec(gs[0])
    ax.plot(E_bend, p(E_bend), color="tab:blue")
    ax.set_ylabel(r"$y_{screen}$ (mm)")
    ax.annotate(
        cam
        + "\n"
        + r"$d_{nom}=$ "
        + "{:0.2f} mm\n".format(d_nom)
        + r"$dy=$"
        + "{:0.2f} mm".format(dy),
        xy=(1, 1),
        xytext=(-3, -3),
        xycoords="axes fraction",
        textcoords="offset points",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )

    ax = fig.add_subplot(gs[1])
    ax.errorbar(y_avg, chargeCal, xerr=y_std, yerr=chargeCalErr, fmt=".", markersize=3)
    ax.set_xlabel(r"$y_{screen}$ (mm)")
    ax.set_ylabel(r"Charge calibration (nC/count)")
    ax.text(
        0.01,
        0.95,
        "{}, Dataset: {}".format(ds.experiment, ds.number),
        transform=ax.transAxes,
    )
    ax.annotate(
        cam,
        xy=(1, 1),
        xytext=(-3, -3),
        xycoords="axes fraction",
        textcoords="offset points",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    plt.show()


# -----------------------------------------------------------------------------------
# Find the calibration
# -----------------------------------------------------------------------------------


def find_calibration(ds, cam, E_0, folder, satThres, satPx, height, crop=None, size=5):
    wf, saturated, img = get_waterfall(ds, cam, satThres, satPx, crop)

    scalarList = ds.getListForPV("TORO:LI20:3163:TMIT")
    toro3163 = ds.getScalar(scalarList, "TORO:LI20:3163:TMIT") * e
    sumCounts = np.sum(wf, axis=0)

    # Find outliers in the dataset
    Q_out = ds.averageByStep(toro3163, outlierRejection=True, returnOutliers=True)[3]
    tC_out = ds.averageByStep(sumCounts, outlierRejection=True, returnOutliers=True)[3]
    # Reject outliers and recalculate
    keep = np.full(ds.N, True)
    keep[Q_out] = False
    keep[tC_out] = False
    Q_avg, Q_std, Q_stdM = ds.averageByStep(toro3163, sel=keep)
    tC_avg, tC_std, tC_stdM = ds.averageByStep(sumCounts, sel=keep)

    chargeCal = Q_avg / tC_avg * 1e9
    chargeCalErr = 1e9 * np.sqrt(
        (Q_std / tC_avg) ** 2 + (Q_avg * tC_std / tC_avg**2) ** 2
    )

    # Find the beam position on the screen
    y = find_beam_center(ds, wf, img, height, size)
    y_avg, y_std, y_stdM = ds.averageByStep(y, sel=keep)
    E_bend = ds.scan_vals
    fit = np.polyfit(E_bend, y_avg, 1)
    p = np.poly1d(fit)
    dy = fit[1]
    d_nom = fit[0] * E_0

    # Plot the data to make sure everything looks okay
    plot_correlations(ds, toro3163, sumCounts, keep, saturated)
    plot_calibration(
        ds, cam, y, y_avg, chargeCal, y_std, chargeCalErr, E_bend, p, d_nom, dy
    )


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [DATASET] [CAM]...",
        description="Generate a report from a rail scan dataset.",
    )
    parser.add_argument(
        "dataset", nargs="1", help="Dataset number to get calibration data from"
    )
    parser.add_argument("cam", nargs="1", help="Camera name to use for calibration")
    return parser


if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()
    ds_number = args.dataset
    cam = args.cam

    experiment = load.srv20_dataset(ds_number)
    ds = dataset.DATASET(experiment, ds_number)
    find_calibration(ds, cam)
