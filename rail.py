import sys
import argparse

import numpy as np
from scipy.special import jv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import load
import image
import dataset
import plot

cam = "RAIL"


def find_center(im, m: float = 0.9):
    """Median filters the given image, then finds the center by using a threshold centroid.

    Args:
        im: Image object to find the center of.
        m: Threshold as a fraction of the max value of the image.
    """
    im.median_filter(size=3)
    thr = m * np.max(im.data)
    im.calculate_center(strategy="mask", threshold=thr)


def find_centers_ds(ds, m: float = 0.9):
    """Finds the center and sumCounts after median filtering every image in the given dataset.

    Args:
        ds: Dataset object with image to find the center of.
        m: Threshold for center finding as a fraction of the max value of the image.

    Returns:
        centers: Array of centers for each image in the dataset.
        sumCounts: Array of sumCounts for each image after median filtering
    """
    centers = np.zeros((ds.N, 2))
    sumCounts = np.zeros(ds.N)
    for i in range(ds.N):
        im = ds.getImage(cam, i)
        find_center(im, m)
        centers[i, :] = im.center
        sumCounts[i] = np.sum(im.data)
    return centers, sumCounts


def reject_outliers(ds, centers, cut):
    """Selects shots that are within the given bound of a line fit through the center position.

    Args:
        ds: Dataset object to reject images from.
        centers: Center position of each image in the dataset.
        cut: Range, +/-, of shots to keep [px].

    Returns:
        selx: Selector that selects the images to keep based on the x criteria.
        sely: Selector that selects the images to keep based on the y criteria.
        px: Polynomial function for the linear fit in x.
        py: Polynomial function for the linear fit in y.
    """
    # Fit lines, select images beyond a certain distance from the line
    z = np.polyfit(ds.x, centers[:, 0], 1)
    px = np.poly1d(z)
    z = np.polyfit(ds.x, centers[:, 1], 1)
    py = np.poly1d(z)

    # Reject images were the center finding is outside of the expected region
    # x direction
    lower = px(ds.x) - cut
    upper = px(ds.x) + cut
    selx = (centers[:, 0] > lower) * (centers[:, 0] < upper)
    lower = py(ds.x) - cut
    upper = py(ds.x) + cut
    sely = (centers[:, 1] > lower) * (centers[:, 1] < upper)
    return selx, sely, px, py


def reject_outliers_sumCounts(ds, sumCounts, lower):
    """Selects shots that have sumCounts above a cutoff.

    Args:
        ds: Dataset object to reject images from.
        sumCounts: Sum counts of each image in the dataset.
        lower: Any shots belkow this threshold will be ignored [counts].

    Returns:
        selsc: Selector that selects the images to keep.
    """
    selsc = sumCounts > lower
    return selsc


def calculate_scale_sim(z, J, ds, maxCounts, sel=None):
    """Calculates the scaling of the simulation output to match the maxCounts.

    Scale is calculated such that the area under the curve is equal for the sim and maxCounts.

    Args:
        z: Z coordinates from the simulation.
        J: On axis fluence at each z point from the simulation.
        ds: Dataset object to the rail images are from.
        maxCounts: Max fluence on the rail camera in counts.

    Returns:
        scale: Multiply the simulation output by scale to match the data.
    """
    dz = z[1] - z[0]
    simN = np.sum(J) * dz * 1e3
    y, y_stdy, y_stdMean = ds.averageByStep(maxCounts, sel=sel)
    dz = ds.scan_vals[1] - ds.scan_vals[0]
    mesN = np.nansum(y) * dz
    scale = mesN / simN
    return scale


def get_max_counts(ds, centers, roi=20):
    """Finds the max counts in a small roi around the beam location.

    Args:
        ds: Dataset object to the rail images are from.
        centers: Center position of each image in the dataset.
        roi: Halfwidth of the roi to find the max in.

    Returns:
        maxCounts: Max fluence on the rail camera in counts.
    """
    N = len(ds.common_index)
    maxCounts = np.zeros(N)
    for i in range(N):
        im = ds.getImage(cam, i)
        cX = int(round(centers[i, 0]))
        cY = int(round(centers[i, 1]))
        data = im.data[cY - roi : cY + roi, cX - roi : cX + roi]
        maxCounts[i] = np.max(data)
    return maxCounts


def get_average_fluence(ds, centers, roi=300, sel=None):
    steps = ds.common_stepIndex
    unique_steps = np.unique(steps)
    inds = np.arange(ds.N)

    num_steps = len(unique_steps)
    fluence = np.zeros((num_steps, 2 * roi, 2 * roi))

    if sel is None:
        sel = np.full(ds.N, True)

    for i, step in enumerate(unique_steps):
        stepSel = (steps == step) * sel
        step_inds = inds[stepSel]
        M = len(step_inds)

        for ind in step_inds:
            img = ds.getImage(cam, ind)
            cX = int(round(centers[ind, 0]))
            cY = int(round(centers[ind, 1]))
            fluence[i, :, :] += img.data[cY - roi : cY + roi, cX - roi : cX + roi]
        fluence[i] /= M
    fluence = np.flip(fluence, axis=0)
    return fluence


def fbsl(x, kr, A, x_0):
    return A * jv(0, (x - x_0) * kr) ** 2


def f2bsl(xy, kr, A, x_0, y_0):
    x = xy[0]
    y = xy[1]
    r = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2)
    return A * jv(0, r * kr) ** 2


def fit_2D_Bessel(image, center, roi, R_b=60e-3, maxfev=1000):
    cX = int(round(center[0]))
    cY = int(round(center[1]))
    x = image.x[cX - roi : cX + roi] - center[0] * image.cal
    y = image.y[cY - roi : cY + roi] - center[1] * image.cal
    X, Y = np.meshgrid(x, y)
    Z = image.data[cY - roi : cY + roi, cX - roi : cX + roi]
    xdata = np.vstack((X.ravel(), Y.ravel()))
    ydata = Z.ravel()
    kr0 = 2.4048 / R_b
    A0 = np.max(Z)
    popt, pcov = curve_fit(f2bsl, xdata, ydata, p0=(kr0, A0, 0.0, 0.0), maxfev=maxfev)
    return f2bsl, popt, x, y, Z


def get_2D_bessel_fit(ds, centers, roi=200, maxfev=1000):
    N = len(ds.common_index)
    fits = np.zeros((N, 4))
    for i in range(N):
        im = ds.getImage(cam, i)
        f2bsl, popt, x, y, Z = fit_2D_Bessel(im, centers[i, :], roi, maxfev=maxfev)
        fits[i, :] = popt
        print("Finished fitting image", i)
    return fits


def fit_1D_bessel(image, center, roi, R_b=60e-3, maxfev=10000):
    cX = int(round(center[0]))
    cY = int(round(center[1]))
    x = image.x[cX - roi : cX + roi] - center[0] * image.cal
    image_data = image.data[cY - roi : cY + roi, cX - roi : cX + roi]
    xdata = image_data[roi, :]
    ydata = image_data[:, roi]
    kr0 = 2.4048 / R_b
    A0 = np.max(xdata)
    poptx, pcov = curve_fit(fbsl, x, xdata, p0=(kr0, A0, 0.0), maxfev=maxfev)
    A0 = np.max(ydata)
    popty, pcov = curve_fit(fbsl, x, ydata, p0=(kr0, A0, 0.0), maxfev=maxfev)
    return poptx, popty


def get_1D_bessel_fit(ds, centers, roi=200):
    N = len(ds.common_index)
    fitsX = np.zeros((N, 3))
    fitsY = np.zeros((N, 3))
    for i in range(N):
        im = ds.getImage(cam, i)
        poptx, popty = fit_1D_bessel(im, centers[i, :], roi)
        fitsX[i, :] = poptx
        fitsY[i, :] = popty
    return fitsX, fitsY


# -----------------------------------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------------------------------


def plot_rejections(ds, centers, cut, sumCounts, lower, folder):
    selx, sely, px, py = reject_outliers(ds, centers, cut)
    selsc = reject_outliers_sumCounts(ds, sumCounts, lower)

    fig = plt.figure(figsize=(4, 3), dpi=300)
    ax = fig.add_subplot()
    ax.plot(ds.x[selx], centers[selx, 0], ".", markersize=1)
    ax.plot(ds.x[~selx], centers[~selx, 0], "x", c="tab:blue", markersize=2)
    ax.plot(ds.x, px(ds.x), "-", c="tab:blue", label="x")
    ax.fill_between(ds.x, px(ds.x) + cut, px(ds.x) - cut, alpha=0.2)
    ax.plot(ds.x[sely], centers[sely, 1], ".", markersize=1)
    ax.plot(ds.x[~sely], centers[~sely, 1], "x", c="tab:red", markersize=2)
    ax.plot(ds.x, py(ds.x), "-", c="tab:red", label="y")
    ax.fill_between(ds.x, py(ds.x) + cut, py(ds.x) - cut, alpha=0.2)
    ax.legend()
    ax.set_ylabel("Center position (px)")
    ax.set_xlabel("Rail motor position (mm)")
    plt.savefig(folder + "{}-01.OutlierRejection.png".format(ds.number))
    # plt.savefig(folder + "01.OutlierRejection.eps")

    fig = plt.figure(figsize=(4, 3), dpi=300)
    ax = fig.add_subplot()
    ax.plot(ds.x, sumCounts[:], ".k", markersize=1)
    ax.plot(ds.x[~selsc], sumCounts[~selsc], "xk", markersize=2)
    ax.plot([ds.x[0], ds.x[-1]], [lower, lower], c="tab:red")
    ax.fill_between(ds.x, lower, 0, alpha=0.2, facecolor="tab:red")
    ax.set_ylabel("Sum counts (counts)")
    ax.set_xlabel("Rail motor position (mm)")
    plt.savefig(folder + "{}-02.OutlierRejectionSumCounts.png".format(ds.number))
    # plt.savefig(folder + "02.OutlierRejectionSumCounts.eps")

    return selx, sely, selsc


def plot_intensity_vs_simulation(fluence, Jxz, ext, ext_sim, scale, ylim=[-0.5, 0.5]):
    xlim = [ext[0], ext[1]]
    roi = int(np.shape(fluence)[1] / 2)

    fig = plt.figure(figsize=(4.2, 3), dpi=300)
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 0.03], hspace=0, wspace=0)

    def plot_slice(ax, yz_slice, ext):
        im = ax.imshow(
            np.transpose(yz_slice),
            aspect="auto",
            extent=ext,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_yticks([-0.25, 0, 0.25])
        return im

    vmin = 0
    vmax = np.nanmax(fluence)
    cmap = plot.cmap_BuW

    ax00 = fig.add_subplot(gs[0, 0])
    im = plot_slice(ax00, fluence[:, roi, :], ext=ext)
    ax00.text(0.02, 0.86, "(a)", transform=ax00.transAxes)
    ax00.text(0.98, 0.86, r"Measured $y=0$ plane", transform=ax00.transAxes, ha="right")
    ax00.set_ylabel(r"$x$ (mm)")

    ax10 = fig.add_subplot(gs[1, 0], sharex=ax00)
    im = plot_slice(ax10, fluence[:, :, roi], ext=ext)
    ax10.text(0.02, 0.86, "(b)", transform=ax10.transAxes)
    ax10.text(0.98, 0.86, r"Measured $x=0$ plane", transform=ax10.transAxes, ha="right")
    ax10.set_ylabel(r"$y$ (mm)")

    ax20 = fig.add_subplot(gs[2, 0], sharex=ax00)
    im = plot_slice(ax20, Jxz * scale, ext=ext_sim)
    ax20.text(0.02, 0.86, "(c)", transform=ax20.transAxes)
    ax20.text(0.98, 0.86, r"Simulation", transform=ax20.transAxes, ha="right")
    ax20.set_xlabel(r"$z$ (mm)")
    ax20.set_ylabel(r"$x$ or $y$ (mm)")

    axcb = fig.add_subplot(gs[:, 1])
    cb1 = fig.colorbar(im, cax=axcb)  # , ticks=np.linspace(-0.010, 0.010, 5))
    cb1.set_label(r"Fluence (Counts)")

    plt.tight_layout()
    plt.setp([a.get_xticklabels() for a in [ax00, ax10]], visible=False)
    return fig


def plot_bessel_lineouts(image_data, x_image, ext, lim=[-1, 1]):
    cmap_RdW = plot.cmap_RdW
    roi = int(np.shape(image_data)[1] / 2)

    fig = plt.figure(figsize=(6.5, 2), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(image_data, extent=ext, cmap=cmap_RdW)
    cb = plt.colorbar(im)
    cb.set_label(r"$I$ (counts)")
    ax0.set_xlim(lim)
    ax0.set_ylim(lim)
    ax0.set_xlabel(r"$x$ (mm)")
    ax0.set_ylabel(r"$y$ (mm)")
    ax0.text(0.04, 0.90, "(a)", transform=ax0.transAxes)

    A0 = np.max(image_data)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(x_image, image_data[roi, :], c="tab:red", label="Data")
    ax1.set_xlabel(r"$x$ (mm)")
    ax1.set_xlim(lim)
    ax1.set_ylim(0, A0 * 1.1)
    ax1.text(0.04, 0.90, "(b)", transform=ax1.transAxes)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(x_image, image_data[:, roi], c="tab:red", label="Data")
    ax2.set_xlabel(r"$y$ (mm)")
    ax2.set_xlim(lim)
    ax2.set_ylim(0, A0 * 1.1)
    ax2.text(0.04, 0.90, "(c)", transform=ax2.transAxes)

    plt.tight_layout()
    return fig, ax0, ax1, ax2


def plot_bessel_fit(image_data, x_image, ext, krx, kry, Ax, Ay, x0, y0, lim=[-1, 1]):
    fig, ax0, ax1, ax2 = plot_bessel_lineouts(image_data, x_image, ext, lim=[-1, 1])

    R_b = 2.4048 / krx
    ax1.plot(x_image, fbsl(x_image, krx, Ax, x0), "k--", label="Fit")
    ax1.text(
        0.04,
        0.8,
        r"$R_b=$" + "{:0.1f}".format(R_b * 1e3) + r"$\,\mathrm{\mu m}$",
        transform=ax1.transAxes,
    )
    ax1.legend(frameon=False, prop={"size": 7}, loc=1, framealpha=1.0)
    ax1.set_ylim(0, Ax * 1.1)

    R_b = 2.4048 / kry
    ax2.plot(x_image, fbsl(x_image, kry, Ay, y0), "k--", label="Fit")
    ax2.text(
        0.04,
        0.8,
        r"$R_b=$" + "{:0.1f}".format(R_b * 1e3) + r"$\,\mathrm{\mu m}$",
        transform=ax2.transAxes,
    )
    ax2.legend(frameon=False, prop={"size": 7}, loc=1, framealpha=1.0)
    ax2.set_ylim(0, Ay * 1.1)
    return fig, ax0, ax1, ax2


# -----------------------------------------------------------------------------------
# Generate the Report
# -----------------------------------------------------------------------------------


def generate_report(ds, folder, cut, lower, roi):
    image.set_calibration(cam, 3.75e-3)
    centers, sumCounts = find_centers_ds(ds)
    # Reject center positions that are unreasonable
    selx, sely, selsc = plot_rejections(ds, centers, cut, sumCounts, lower, folder)
    sel = selx * sely * selsc
    text = input("Data pruning completed, continue (y/n)?")
    if text != "y":
        return
    # Possible to add sumCounts outlier rejection if required
    maxCounts = get_max_counts(ds, centers, roi=20)
    fluence = get_average_fluence(ds, centers, roi=roi, sel=sel)
    return centers, sel, maxCounts, fluence


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [DATASET]...",
        description="Generate a report from a rail scan dataset.",
    )
    parser.add_argument(
        "dataset", nargs="1", help="Dataset number to generate a report from"
    )
    return parser


if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()
    ds_number = args.dataset

    experiment = load.srv20_dataset(ds_number)
    ds = dataset.DATASET(experiment, ds_number)
    # generate_report(ds)
