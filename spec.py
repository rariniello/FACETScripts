import math
import json

import numpy as np
from scipy import integrate
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from image import IMAGE, Elog, DAQ
import image
import plot
import analysis as an
from scipy.constants import physical_constants

c = physical_constants["speed of light in vacuum"][0]
e = physical_constants["elementary charge"][0]
me = physical_constants["electron mass"][0]
eps_0 = physical_constants["vacuum electric permittivity"][0]


def forward(y, d_nom, E_bend, dy):
    return d_nom * E_bend / (dy - y)


def inverse(E, d_nom, E_bend, dy):
    return dy - d_nom * E_bend / E


def loadCalibration(path):
    with open(path) as f:
        d = json.load(f)
    return d


def plot_calibration_comparison(ds, cal):
    Q0D = ds.getScalar("nonBSA_List_S20Magnets", "LI20:LGPS:3141:BACT")[0]
    Q1D = ds.getScalar("nonBSA_List_S20Magnets", "LI20:LGPS:3261:BACT")[0]
    Q2D = ds.getScalar("nonBSA_List_S20Magnets", "LI20:LGPS:3091:BACT")[0]

    BPM3156_Y = np.nanmean(ds.getScalar("BSA_List_S20", "BPMS:LI20:3156:Y"))
    BPM3218_Y = np.nanmean(ds.getScalar("BSA_List_S20", "BPMS:LI20:3218:Y"))
    BPM3265_Y = np.nanmean(ds.getScalar("BSA_List_S20", "BPMS:LI20:3265:Y"))
    BPM3315_Y = np.nanmean(ds.getScalar("BSA_List_S20", "BPMS:LI20:3315:Y"))

    labels_magnets = ["Q0D", "Q1D", "Q2D"]
    cal_magnets = [cal["Q0D"], cal["Q1D"], cal["Q2D"]]
    act_magnets = [Q0D, Q1D, Q2D]
    labels_bpms = ["BPM3156", "BPM3218", "BPM3265", "BPM3315"]
    cal_bpms = [
        cal["BPM3156"][1],
        cal["BPM3218"][1],
        cal["BPM3265"][1],
        cal["BPM3315"][1],
    ]
    act_bpms = [BPM3156_Y, BPM3218_Y, BPM3265_Y, BPM3315_Y]

    width = 0.3  # the width of the bars

    fig = plt.figure(figsize=(4, 3), dpi=300)
    gs = gridspec.GridSpec(1, 2, wspace=0.0, width_ratios=(3, 4))
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(labels_magnets))
    b1 = ax.bar(x + 0.5 * width, act_magnets, width, color="tab:blue")
    b2 = ax.bar(x + 1.5 * width, cal_magnets, width, color="tab:green")
    ax.set_xticks(x + width, labels_magnets)

    ax.set_ylim(-160, 160)
    ax.plot([-0.5, x[-1] + 1], [0.0, 0.0], "k")
    ax.set_xlim([-0.5, x[-1] + 1])
    ax.set_ylabel("Quad setting")
    ax.bar_label(b1, fmt="%.2f", fontsize="6", rotation="vertical", padding=2)
    ax.bar_label(b2, fmt="%.2f", fontsize="6", rotation="vertical", padding=2)

    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(labels_bpms))
    b1 = ax2.bar(
        x + 0.5 * width,
        act_bpms,
        width,
        color="tab:blue",
        label="Dataset {}".format(ds.number),
    )
    b2 = ax2.bar(
        x + 1.5 * width,
        cal_bpms,
        width,
        color="tab:green",
        label="Calibration {}".format(cal["dataset"]),
    )
    ax2.set_xticks(x + width, labels_bpms)

    ax2.set_ylim(-4, 4)
    ax2.plot([-0.5, x[-1] + 1], [0.0, 0.0], "k")
    ax2.set_xlim([-0.5, x[-1] + 1])
    ax2.set_ylabel("BPM offset (mm)")
    ax2.bar_label(b1, fmt="%.2f", fontsize="6", rotation="vertical", padding=2)
    ax2.bar_label(b2, fmt="%.3f", fontsize="6", rotation="vertical", padding=2)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.legend()
    plt.tight_layout()
    return fig, ax, ax2


def plot_cher_background(ds, crop):
    cam = "CHER"
    bkgd = ds.getImageBackground(cam)
    img = ds.getImage(cam, 0)
    img.data[:, :] = 0.0
    img.subtract_background(-1 * bkgd)
    fig, ax, im, cb, ext = img.plot_image(cal=False)
    im.set_clim(0, 200)
    ax.plot(
        [crop[0], crop[2], crop[2], crop[0], crop[0]],
        [crop[1], crop[1], crop[3], crop[3], crop[1]],
        "w",
    )
    plt.tight_layout()
    return fig, ax


def calcCharge(E, spectrum):
    Q = integrate.trapezoid(np.flip(spectrum), x=np.flip(E))
    return Q


def calcChargeError(E, spectrum, chargeCal, chargeCalErr):
    QErr = integrate.trapezoid(
        np.flip(spectrum * chargeCalErr / chargeCal), x=np.flip(E)
    )
    return QErr


def calcMinMaxEnergy(E, spectrum, threshold, constant=False):
    projMax = np.max(spectrum)
    minThres = threshold * projMax
    if constant:
        minThres = threshold
    limit = 2.5e-3
    if minThres < limit:
        minThres = limit
    t = spectrum - minThres
    try:
        roots = an.find_roots(E, t)
    except:
        minEnergy = 0.0
        maxEnergy = 0.0
        if minThres < spectrum[-1]:
            minEnergy = E[-1]
        if minThres < spectrum[0]:
            maxEnergy = E[0]
        return minEnergy, maxEnergy
    minEnergy = roots[-1]
    maxEnergy = roots[0]
    if minThres < spectrum[-1]:
        minEnergy = E[-1]
    if minThres < spectrum[0]:
        maxEnergy = E[0]
    return minEnergy, maxEnergy


def calcVisibleEnergy(E, spectrum):
    E_vis = integrate.trapezoid(np.flip(spectrum * E), x=np.flip(E))
    return E_vis


def calcVisibleEnergyError(
    E, spectrum, chargeCal, chargeCalErr, y, d_nom, d_nom_err, dy, dy_err
):
    E_vis = integrate.trapezoid(np.flip(spectrum * E), x=np.flip(E))
    err_Q = integrate.trapezoid(
        np.flip(spectrum * E * chargeCalErr / chargeCal), x=np.flip(E)
    )
    err_dnom = E_vis * d_nom_err / d_nom
    err_dy = (
        integrate.trapezoid(np.flip(spectrum * E / (dy - y)), x=np.flip(E)) * dy_err
    )
    E_vis_err = np.sqrt(err_Q**2 + err_dnom**2 + err_dy**2)
    return E_vis_err


def calc_spectrum_parameters(cher_ds, bpm3156, E_2445):
    # Other things to calculate: non-participating charge
    N = cher_ds.N
    Q = np.zeros(N)
    QErr = np.zeros(N)
    E_vis = np.zeros(N)
    E_vis_err = np.zeros(N)
    minE = np.zeros(N)
    maxE = np.zeros(N)
    E_i = bpm3156 * 1e9 * E_2445

    ds = cher_ds.dataset
    specImg = cher_ds.getSpectrumImage(0)
    E, spectrum, dE = specImg.getSpectrum()
    y = np.flip(specImg.image.y)

    def f(x, x0, sigma, A):
        return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    for i in range(N):
        spectrum = cher_ds.wf[:, i]
        Q[i] = calcCharge(E, spectrum)
        QErr[i] = calcChargeError(E, spectrum, cher_ds.chargeCal, cher_ds.chargeCalErr)
        E_vis[i] = calcVisibleEnergy(E, spectrum)
        E_vis_err[i] = calcVisibleEnergyError(
            E,
            spectrum,
            cher_ds.chargeCal,
            cher_ds.chargeCalErr,
            y,
            cher_ds.d_nom,
            cher_ds.d_nom_err,
            cher_ds.dy,
            cher_ds.dy_err,
        )
        proj = savgol_filter(spectrum, 25, 2)
        minE[i], maxE[i] = calcMinMaxEnergy(E, proj, 0.08)
        if minE[i] == E[-1]:
            minE[i] = 0.0

    Q_mis = bpm3156 * 1e9 - Q
    E_mis_max = Q_mis * E_2445
    E_mis_min = Q_mis * minE
    Eloss_min = E_i - E_vis - E_mis_max
    Eloss_max = E_i - E_vis - E_mis_min
    eta_min = Eloss_min / E_i
    eta_max = Eloss_max / E_i

    ds.PVsInList("BSA_List_S14")
    S14BLEN = ds.getScalar("BSA_List_S14", "BLEN:LI14:888:BRAW")

    data = {
        "Q": Q,
        "QErr": QErr,
        "E_vis": E_vis,
        "E_vis_err": E_vis_err,
        "minE": minE,
        "maxE": maxE,
        "E_i": E_i,
        "Q_mis": Q_mis,
        "E_mis_max": E_mis_max,
        "E_mis_min": E_mis_min,
        "Eloss_min": Eloss_min,
        "Eloss_max": Eloss_max,
        "eta_min": eta_min,
        "eta_max": eta_max,
        "S14BLEN": S14BLEN,
        "bpm3156": bpm3156,
    }
    return data


class SPEC:
    def __init__(
        self,
        image,
        E_bend,
        d_nom,
        d_nom_err,
        dy,
        dy_err,
        chargeCal,
        chargeCalErr,
    ):
        self.image = image
        self.d_nom = d_nom
        self.d_nom_err = d_nom_err
        self.E_bend = E_bend
        self.dy = dy
        self.dy_err = dy_err
        self.chargeCal = chargeCal
        self.chargeCalErr = chargeCalErr
        self.E = None
        self.dE = None
        self.spectrum = None

    def forward(self, y):
        return forward(y, self.d_nom, self.E_bend, self.dy)

    def inverse(self, E):
        return inverse(E, self.d_nom, self.E_bend, self.dy)

    def getSpectrum(self):
        if self.spectrum is not None:
            return self.E, self.spectrum, self.dE
        y = np.flip(self.image.y)
        self.E = self.forward(y)
        N = np.sum(self.image.data, axis=1)
        E_bin_s = self.forward(y - 0.5 * self.image.cal)
        E_bin_e = self.forward(y + 0.5 * self.image.cal)
        self.dE = E_bin_e - E_bin_s
        dNdE = N / self.dE
        dQdE = dNdE * self.chargeCal
        self.spectrum = dQdE
        return self.E, self.spectrum, self.dE

    def getCharge(self):
        self.getSpectrum()
        self.Q = calcCharge(self.E, self.spectrum)
        return self.Q

    def plot_image(self, cmap=plot.roar, metadata=True):
        fig, ax, im, cb, ext = self.image.plot_image(
            cal=True, cmap=cmap, metadata=metadata
        )
        # TODO Calculate width based on figure width
        fig.set_size_inches(2.85, 6.1)
        E_ticks = np.arange(
            math.ceil(self.forward(ext[3])), math.floor(self.forward(ext[2])) + 1
        )
        tick_locs = [ext[2] - self.inverse(E) for E in E_ticks]
        tick_lbls = ["{:0.1f}".format(E) for E in E_ticks]
        ax.set_ylabel("Energy (GeV)")
        plt.yticks(tick_locs, tick_lbls)
        ax.tick_params(color="k")
        ax.spines["bottom"].set_color("k")
        ax.spines["top"].set_color("k")
        ax.spines["left"].set_color("k")
        ax.spines["right"].set_color("k")
        if metadata:
            ax.stepText.set_color("k")
            ax.datasetText.set_color("k")
            ax.metadataText.set_color("k")
            self.append_metadata_text(ax.metadataText)
        return fig, ax, im, cb, ext

    def plot_spectrum(self, metadata=True):
        self.getSpectrum()
        fig = plt.figure(figsize=(2.85, 2), dpi=300)
        ax = fig.add_subplot()
        ax.plot(self.E, self.spectrum)
        ax.set_ylabel(r"$dQ/dE$ (nC/GeV)")
        ax.set_xlabel(r"Energy (GeV)")
        ax.set_xlim(self.E[-1], self.E[0])
        ax.set_ylim(0, 1.2 * np.max(self.spectrum))
        if metadata:
            self.image.plot_dataset_text(ax)
            # self.image.plot_metadata_text(ax)
            ax.stepText.set_color("k")
            ax.datasetText.set_color("k")
            # ax.metadataText.set_color("k")
        return fig, ax

    def append_metadata_text(self, metadataText):
        text = metadataText.get_text()
        addText = (
            r"$dy$:      "
            + "{:0.2f}\n".format(self.dy)
            + r"$d_{nom}$:   "
            + "{:0.2f}\n".format(self.d_nom)
            + r"$E_{bend}$:  "
            + "{:0.1f}GeV\n".format(self.E_bend)
        )
        text = addText + text
        metadataText.set_text(text)


class SPEC_DS:
    def __init__(
        self,
        dataset,
        cam,
        d_nom,
        d_nom_err,
        dy,
        dy_err,
        chargeCal,
        chargeCalErr,
        E_bend_default=10.0,
        crop=None,
        subtract_background=True,
        median_filter=None,
    ):
        """
        Args:
            E_bend: Default E_bend if the scalar isn't present in the dataset.
            crop: The crop rectangle, as a (left, upper, right, lower) tuple.
        """
        self.cam = cam
        self.dataset = dataset
        self.d_nom = d_nom
        self.d_nom_err = d_nom_err
        self.dy = dy
        self.dy_err = dy_err
        self.chargeCal = chargeCal
        self.chargeCalErr = chargeCalErr
        self.E_bend_default = E_bend_default
        self.crop = crop
        self.N = len(dataset.common_index)
        self.img = dataset.getImage(cam, 0)
        self.initial_height = self.img.height
        if self.crop is not None:
            self.img.crop(self.crop)
        self.wf = None
        self.wf_linear = None
        self.subtract_background = subtract_background
        self.median_filter = median_filter

        self.adjust_dy()

        ds = self.dataset
        list = ds.getListForPV("LI20:LGPS:3330:BACT")
        self.E_bend = ds.getScalar(list, "LI20:LGPS:3330:BACT")[0]

        self._warned = False

    def forward(self, y):
        return forward(y, self.d_nom, self.E_bend, self.dy)

    def inverse(self, E):
        return inverse(E, self.d_nom, self.E_bend, self.dy)

    def getSpectrumImage(self, ind):
        ds = self.dataset
        img = ds.getImage(self.cam, ind)
        if self.subtract_background:
            bkgd = ds.getImageBackground(self.cam)
            img.subtract_background(bkgd)
        list = ds.getListForPV("LI20:LGPS:3330:BACT")
        if list is not None:
            E_bend = ds.getScalar(list, "LI20:LGPS:3330:BACT")[ind]
        else:
            E_bend = self.E_bend_default
            print(
                "Dipole setting not found in scalar lists, defaulting to {:0.1f}GeV.".format(
                    E_bend
                )
            )
        if np.isnan(E_bend):
            E_bend = self.E_bend_default
            if not self._warned:
                print(
                    "Dipole setting is nan in scalar lists, defaulting to {:0.1f}GeV.".format(
                        E_bend
                    )
                )
                self._warned = True
        if self.crop is not None:
            img.crop(self.crop)
        if self.median_filter is not None:
            img.median_filter(self.median_filter)
        specImg = SPEC(
            img,
            E_bend,
            self.d_nom,
            self.d_nom_err,
            self.dy,
            self.dy_err,
            self.chargeCal,
            self.chargeCalErr,
        )
        return specImg

    def get_waterfall(self):
        if self.wf is not None:
            return self.wf
        N = self.N
        M = self.img.height
        wf = np.zeros((M, N))

        for i in range(N):
            specImg = self.getSpectrumImage(i)
            specImg.getSpectrum()
            wf[:, i] = specImg.spectrum
        self.wf = wf
        return wf

    def get_linear_waterfall(self):
        if self.wf_linear is not None:
            return self.wf_linear

        self.get_waterfall()
        N = self.N
        M = self.img.height
        wf_linear = np.zeros((M, N))

        specImg = self.getSpectrumImage(0)
        E, spectrum, dE = specImg.getSpectrum()
        E_linear = np.linspace(E[0], E[-1], M)

        for i in range(N):
            f = interpolate.interp1d(E, self.wf[:, i])
            wf_linear[:, i] = f(E_linear)
        self.wf_linear = wf_linear
        self.E_linear = E_linear
        return wf_linear

    def adjust_dy(self):
        # If the crop region doesn't reach the bottom of the screen then need to adjust dy to compensate
        if self.initial_height != self.crop[3]:
            self.dy -= (self.initial_height - self.crop[3]) * self.img.cal

    def plot_waterfall(self):
        self.get_waterfall()
        ext = self.img.get_ext()
        ext[0] = -0.5
        ext[1] = self.N - 0.5

        fig = plt.figure(figsize=(6, 4), dpi=300)
        ax = fig.add_subplot()
        ax.dataset_text = ax.text(
            0.01,
            1.02,
            "{}, Dataset: {}".format(self.dataset.experiment, self.dataset.number),
            transform=ax.transAxes,
        )
        im = ax.imshow(
            self.wf,
            aspect="auto",
            cmap=plot.cmap_W_Viridis,
            interpolation="none",
            extent=ext,
        )
        cb = fig.colorbar(im)
        cb.set_label(r"$dQ/dE$ (nC/GeV)")
        im.set_clim(0, np.max(self.wf))
        ax.set_xlabel("Shot number")

        E_ticks = np.arange(
            math.ceil(self.forward(ext[3])),
            math.floor(self.forward(ext[2])) + 1,
        )
        tick_locs = [ext[2] - self.inverse(E) for E in E_ticks]
        tick_lbls = ["{:0.1f}".format(E) for E in E_ticks]
        ax.set_ylabel("Energy (GeV)")
        plt.yticks(tick_locs, tick_lbls)
        return fig, ax, im

    def plot_linear_waterfall(self):
        self.get_linear_waterfall()
        dE = self.E_linear[1] - self.E_linear[0]
        ext = np.zeros(4)
        ext[0] = -0.5
        ext[1] = self.N - 0.5
        ext[3] = self.E_linear[0] - 0.5 * dE
        ext[2] = self.E_linear[-1] + 0.5 * dE

        fig = plt.figure(figsize=(6, 4), dpi=300)
        ax = fig.add_subplot()
        ax.dataset_text = ax.text(
            0.01,
            1.02,
            "{}, Dataset: {}".format(self.dataset.experiment, self.dataset.number),
            transform=ax.transAxes,
        )
        im = ax.imshow(
            self.wf_linear,
            aspect="auto",
            cmap=plot.cmap_W_Viridis,
            interpolation="none",
            extent=ext,
        )
        cb = fig.colorbar(im)
        cb.set_label(r"$dQ/dE$ (nC/GeV)")
        im.set_clim(0, np.max(self.wf))
        ax.set_xlabel("Shot number")
        ax.set_ylabel("Energy (GeV)")
        return fig, ax, im
