import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

import image


def plotCalibrationRectangle(image_cal, box, angle, cmax=None, name=None):
    image_cal.rotate(angle)
    fig, ax, im, cb, ext = image_cal.plot_image(cal=False)
    x_s = box[0]
    x_e = box[1]
    y_s = box[2]
    y_e = box[3]
    ax.plot([x_s, x_s], [y_s, y_e], "w")
    ax.plot([x_e, x_e], [y_s, y_e], "w")
    ax.plot([x_s, x_e], [y_s, y_s], "w")
    ax.plot([x_s, x_e], [y_e, y_e], "w")
    if cmax is not None:
        im.set_clim(0, cmax)
    if name is not None:
        plt.savefig(name)
    plt.show()


def getCalibrationThreeLine(cal_image, box, thr, lp_per_mm, orientation):
    """Calculates the mm/px from an image with a three bar calibration target.

    Args:
        image: Image object for the calibration image.
        box: The sub-region of the image to take the calibration from (x_start, e_end, y_start, y_end) [px].
        thr: Threshold value to find rising edge and falling edge intersections at.
        lp_per_mm: Line pairs per mm of the three bars.
        orientation: Orientation of the bars, either 'horizontal' or 'vertical'.

    Returns:
        cal: Pixel calibration of the image [mm/px].
        cal_std: Standard deviation of the calibration for all edge pairs [mm/px].
        cal_line: Line out across the bars from averaging over one axis of the image sub-region.
        root: X location of the intersect between the threshold line and the edges.
        root_y:
    """
    # Calulate spacing between pairs of falling edges and pairs of rising edges
    if orientation == "horizontal":
        cal_line = np.average(cal_image.data[box[2] : box[3], box[0] : box[1]], axis=1)
    elif orientation == "vertical":
        cal_line = np.average(cal_image.data[box[2] : box[3], box[0] : box[1]], axis=0)
    else:
        raise ValueError(
            "Orientation must be either 'horizontal' or 'vertical'. Instead {} was passed.".format(
                orientation
            )
        )
    x = np.arange(len(cal_line))
    spline = UnivariateSpline(x, cal_line - thr, s=0)
    roots = spline.roots()
    if len(roots) < 6:
        print(
            "Not enough roots for calibration, check the threshold value is appropriate."
        )
        return None, None, cal_line, roots
    dist = np.array(
        [
            roots[2] - roots[0],
            roots[4] - roots[2],
            roots[3] - roots[1],
            roots[5] - roots[3],
        ]
    )
    cal_array = 1 / lp_per_mm / dist
    cal = np.average(cal_array)
    cal_std = np.std(cal_array) / np.sqrt(len(cal_array))
    return cal, cal_std, cal_line, roots


def plotCalibrationThreeLine(
    thr, orientation, cal, cal_std, cal_line, roots, name=None
):
    x = np.arange(len(cal_line))
    fig = plt.figure(figsize=(5, 2), dpi=150)
    ax = fig.add_subplot()
    ax.plot(x, cal_line)
    ax.plot([x[0], x[-1]], [thr, thr])
    ax.plot(roots, np.ones(len(roots)) * thr, "k.", markersize=3)
    if orientation == "horizontal":
        plt.xlabel(r"$y$ (px)")
    elif orientation == "vertical":
        plt.xlabel(r"$x$ (px)")
    plt.ylabel("Counts")
    plt.text(
        0.02,
        0.9,
        r"{:0.2f}$\pm${:0.2f} um/px".format(cal * 1e3, cal_std * 1e3),
        transform=ax.transAxes,
    )
    if name is not None:
        plt.savefig(name)
    plt.show()


def getCalibrationRonchiRuling(cal_image, box, thr, lp_per_mm, orientation):
    """Calculates the mm/px from an image with a ronchi ruling calibration target.

    Args:
        image: Image object for the calibration image.
        box: The sub-region of the image to take the calibration from (x_start, e_end, y_start, y_end) [px].
        thr: Threshold value to find rising edge and falling edge intersections at.
        lp_per_mm: Line pairs per mm of the three bars.
        orientation: Orientation of the bars, either 'horizontal' or 'vertical'.

    Returns:
        cal: Pixel calibration of the image [mm/px].
        cal_std: Standard deviation of the calibration for all edge pairs [mm/px].
        cal_line: Line out across the bars from averaging over one axis of the image sub-region.
        root: X location of the intersect between the threshold line and the edges.
        root_y:
    """
    # Calulate spacing between pairs of falling edges and pairs of rising edges
    if orientation == "horizontal":
        cal_line = np.average(cal_image.data[box[2] : box[3], box[0] : box[1]], axis=1)
    elif orientation == "vertical":
        cal_line = np.average(cal_image.data[box[2] : box[3], box[0] : box[1]], axis=0)
    else:
        raise ValueError(
            "Orientation must be either 'horizontal' or 'vertical'. Instead {} was passed.".format(
                orientation
            )
        )
    x = np.arange(len(cal_line))
    spline = UnivariateSpline(x, cal_line - thr, s=0)
    roots = spline.roots()
    dist = roots[2:] - roots[:-2]
    cal_array = 1 / lp_per_mm / dist
    cal = np.average(cal_array)
    cal_std = np.std(cal_array) / np.sqrt(len(cal_array))

    # Do it by fitting
    period_roots = np.average(dist)

    def func(x, a, w, s_x, s_y):
        return a * np.cos(w * (x - s_x)) + s_y

    popt, pcov = curve_fit(
        func, x, cal_line, p0=(500, 2 * np.pi / period_roots, 0, 500)
    )
    period_fit = 2 * np.pi / popt[1]
    cal_fitting = 1 / lp_per_mm / period_fit
    return cal, cal_std, cal_line, roots, cal_fitting, func, popt


def plotCalibrationRonchiRuling(
    thr, orientation, cal, cal_std, cal_line, roots, cal_fitting, func, popt, name=None
):
    x = np.arange(len(cal_line))
    x2 = np.linspace(x[0], x[-1], 1000)
    fig = plt.figure(figsize=(5, 2), dpi=150)
    ax = fig.add_subplot()
    ax.plot(x, cal_line)
    ax.plot([x[0], x[-1]], [thr, thr])
    ax.plot(roots, np.ones(len(roots)) * thr, "k.", markersize=3)
    ax.plot(x2, func(x2, *popt))
    if orientation == "horizontal":
        plt.xlabel(r"$y$ (px)")
    elif orientation == "vertical":
        plt.xlabel(r"$x$ (px)")
    plt.ylabel("Counts")
    plt.text(
        0.02,
        0.9,
        r"{:0.2f}$\pm${:0.2f} um/px".format(cal * 1e3, cal_std * 1e3),
        transform=ax.transAxes,
    )
    plt.text(
        0.02,
        0.8,
        r"{:0.2f}$\pm${:0.2f} um/px from fit".format(cal_fitting * 1e3, cal_std * 1e3),
        transform=ax.transAxes,
    )
    if name is not None:
        plt.savefig(name)
    plt.show()


def scanRotation(
    camera,
    date,
    time,
    angle,
    box,
    thr,
    lp_per_mm,
    orientation,
    func=getCalibrationThreeLine,
):
    N = 100
    angles = np.linspace(angle - 2.5, angle + 2.5, N)
    cal = np.zeros(N)
    for i in range(N):
        cal_image = image.Elog(camera, date, time)
        cal_image.rotate(angles[i])
        cal[i] = func(cal_image, box, thr, lp_per_mm, orientation)[0]
    return angles, cal


def scanThreshold(
    camera,
    date,
    time,
    angle,
    box,
    thr,
    lp_per_mm,
    orientation,
    func=getCalibrationThreeLine,
    N=400,
):
    thrs = np.linspace(thr - N / 2, thr + N / 2, N)
    cal = np.zeros(N)
    cal_image = image.Elog(camera, date, time)
    cal_image.rotate(angle)
    for i in range(N):
        cal[i] = func(cal_image, box, thrs[i], lp_per_mm, orientation)[0]
    return thrs, cal


def plotScans(
    angles, cal_angles, thrs, cal_thrs, angle, cal, cal_std, thr, N=400, name=None
):
    fig = plt.figure(figsize=(6, 2), dpi=150)
    ax = fig.add_subplot(121)
    ax.plot(angles, cal_angles)
    ax.plot([angle - 2.5, angle + 2.5], [cal, cal], "k")
    ax.plot([angle - 2.5, angle + 2.5], [cal + cal_std, cal + cal_std], "k--")
    ax.plot([angle - 2.5, angle + 2.5], [cal - cal_std, cal - cal_std], "k--")
    ax.set_xlabel("Image Rotation (deg)")
    ax.set_ylabel("Calibration (mm/px)")
    min = None
    max = None
    if np.max(cal_angles) > cal + 4 * cal_std:
        max = cal + 2 * cal_std
    if np.min(cal_angles) < cal - 4 * cal_std:
        min = cal - 2 * cal_std
    if max is not None or min is not None:
        if max is None:
            max = cal + 1.1 * cal_std
        if min is None:
            min = cal - 1.1 * cal_std
        ax.set_ylim(min, max)

    ax = fig.add_subplot(122)
    ax.plot(thrs, cal_thrs)
    ax.plot([thr - N / 2, thr + N / 2], [cal, cal], "k")
    ax.plot([thr - N / 2, thr + N / 2], [cal + cal_std, cal + cal_std], "k--")
    ax.plot([thr - N / 2, thr + N / 2], [cal - cal_std, cal - cal_std], "k--")
    ax.set_xlabel("Crossing Level (Counts)")
    ax.set_ylabel("Calibration (mm/px)")
    min = None
    max = None
    if np.max(cal_thrs) > cal + 4 * cal_std:
        max = cal + 2 * cal_std
    if np.min(cal_thrs) < cal - 4 * cal_std:
        min = cal - 2 * cal_std
    if max is not None or min is not None:
        if max is None:
            max = cal + 1.1 * cal_std
        if min is None:
            min = cal - 1.1 * cal_std
        ax.set_ylim(min, max)
    plt.tight_layout()
    if name is not None:
        plt.savefig(name)
    plt.show()
