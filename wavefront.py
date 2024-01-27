import numpy as np
from PIL import Image
from scipy import ndimage
import base64
import ast
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from scipy.optimize import curve_fit

# import libtiff
import glob
import os
import re
import warnings

PATH = ""

RdBu = mpl.cm.get_cmap("RdBu")
cl = np.zeros((11, 3))
for i in range(11):
    cl[i, :] = [
        RdBu._segmentdata["red"][i, 1],
        RdBu._segmentdata["green"][i, 1],
        RdBu._segmentdata["blue"][i, 1],
    ]
cl[5, :] = [1.0, 1.0, 1.0]
cmap_RdWBu = colors.LinearSegmentedColormap.from_list("RdWBu", cl)
cmap_BuW = colors.LinearSegmentedColormap.from_list("RdWBu", cl[5:])
cmap_RdW = colors.LinearSegmentedColormap.from_list("RdWBu", np.flip(cl[:6], axis=0))


def parse_line(line, rexp):
    """Regex search against the defined lines, return the first match."""
    for key, rex in rexp.items():
        match = rex.search(line)
        if match:
            return key, match
    return None, None


class PHASE:
    rexp = {
        "sensor": re.compile(
            r"ASO Size : (?P<X>[0-9]{2}) x (?P<Y>[0-9]{2}); ASO Step : (?P<dx>.*) um x (?P<dy>.*) um"
        ),
    }

    # Initialization functions -------------------------------------------------
    # --------------------------------------------------------------------------
    def __init__(self, filename):
        self.filename = filename
        self.phase = self.load_phase()
        self.data = self.phase
        self.load_meta()
        self.xp = np.arange(0, self.width, 1)
        self.yp = np.arange(0, self.height, 1)
        self.x = self.dx * self.xp
        self.y = self.dy * self.yp

    def load_phase(self) -> np.ndarray:
        phase = np.loadtxt(self.filename, dtype="S", delimiter="\t", skiprows=8)
        sel = phase == b""
        phase[sel] = "Nan"
        return np.array(phase, dtype="float")

    def load_meta(self) -> dict:
        with open(self.filename, "r") as f:
            for line in f:
                key, match = parse_line(line, self.rexp)
                if key == "sensor":
                    self.width = int(match.group("X"))
                    self.height = int(match.group("Y"))
                    self.dx = float(match.group("dx"))
                    self.dy = float(match.group("dy"))

    # Analysis functions -------------------------------------------------------
    # --------------------------------------------------------------------------
    def remove_tilt_focus(self, poly):
        tilt_h = poly.polynomial_phase_haso(1)
        tilt_v = poly.polynomial_phase_haso(2)
        focus = poly.polynomial_phase_haso(3)
        return self.phase - tilt_h - tilt_v - focus

    # Visualization functions --------------------------------------------------
    # --------------------------------------------------------------------------
    def plot_phase(self, phase=None):
        fig = plt.figure(figsize=(5.4, 3), dpi=150)
        ax = fig.add_subplot(111)
        fig, ax, im, cb = self.phase_axis(fig, ax, phase)
        return fig, ax, im, cb

    def phase_axis(self, fig, ax, phase=None):
        if phase is None:
            phase = self.phase
        im_max = np.nanmax(phase)
        im_min = np.nanmin(phase)
        cen = 0.5 * (im_max - im_min) + im_min
        im = ax.imshow(phase - cen, cmap="RdYlBu")  # , vmin=-im_max, vmax=im_max)
        cb = fig.colorbar(im)
        cb.set_label(r"Phase ($\mathrm{\mu m}$)")
        ax.set_xlabel(r"$x$ (px)")
        ax.set_ylabel(r"$y$ (px)")
        ax.set_xlim(-9.5, phase.shape[1] - 0.5)
        ax.set_ylim(phase.shape[0] - 0.5, -0.5)
        ax.text(
            -9,
            2,
            r"P-V Phase: %0.3f $\mathrm{\mu m}$" % (im_max - im_min),
        )
        ax.text(
            -9,
            4,
            r"RMS Phase: %0.3f $\mathrm{\mu m}$"
            % (np.std(phase, where=~np.isnan(phase))),
        )
        return fig, ax, im, cb

    def phase_error_axis(
        self, fig, ax, poly, phase=None, subtracted=False, sub_f=False
    ):
        if phase is None:
            phase = self.phase
        tilt_h = poly.polynomial_phase_haso(1)
        tilt_v = poly.polynomial_phase_haso(2)
        focus = poly.polynomial_phase_haso(3)
        if subtracted:
            fig, ax, im, cb = self.phase_axis(fig, ax, phase)
        elif sub_f:
            fig, ax, im, cb = self.phase_axis(fig, ax, phase - tilt_h - tilt_v)
        else:
            fig, ax, im, cb = self.phase_axis(fig, ax, phase - tilt_h - tilt_v - focus)
        ax.text(
            -9,
            6,
            r"Tilt H: %0.3f $\mathrm{\mu m}$" % (poly.coefficients[0]),
        )
        ax.text(-9, 8, r"Tilt V: %0.3f $\mathrm{\mu m}$" % (poly.coefficients[1]))
        ax.text(-9, 10, r"Defocus : %0.3f $\mathrm{\mu m}$" % (poly.coefficients[2]))
        return fig, ax, im, cb

    def plot_phase_error(self, poly, phase=None, subtracted=False, sub_f=False):
        fig = plt.figure(figsize=(5.4, 3), dpi=150)
        ax = fig.add_subplot(111)
        fig, ax, im, cb = self.phase_error_axis(fig, ax, poly, phase, subtracted, sub_f)
        return fig, ax, im, cb


class INTENSITY:
    rexp = {
        "sensor": re.compile(
            r"ASO Size : (?P<X>[0-9]{2}) x (?P<Y>[0-9]{2}); ASO Step : (?P<dx>.*) um x (?P<dy>.*) um"
        ),
    }

    # Initialization functions ------------------------------------------------
    # --------------------------------------------------------------------------
    def __init__(self, filename):
        self.filename = filename
        self.intensity = self.load_intensity()
        self.data = self.intensity
        self.load_meta()
        self.xp = np.arange(0, self.width, 1)
        self.yp = np.arange(0, self.height, 1)
        self.x = self.dx * self.xp
        self.y = self.dy * self.yp

    def load_intensity(self) -> object:
        intensity = np.loadtxt(self.filename, dtype="S", delimiter="\t", skiprows=6)
        sel = intensity == b""
        intensity[sel] = "Nan"
        return np.array(intensity, dtype="float")

    def load_meta(self) -> dict:
        with open(self.filename, "r") as f:
            for line in f:
                key, match = parse_line(line, self.rexp)
                if key == "sensor":
                    self.width = int(match.group("X"))
                    self.height = int(match.group("Y"))
                    self.dx = float(match.group("dx"))
                    self.dy = float(match.group("dy"))

    # Visualization functions -------------------------------------------------
    # --------------------------------------------------------------------------
    def plot_intensity(self, intensity=None):
        if intensity is None:
            intensity = self.intensity
        im_max = np.nanmax(intensity)
        im_min = np.nanmin(intensity)
        fig = plt.figure(figsize=(4.85, 3), dpi=300)
        ax = plt.subplot()
        im = ax.imshow(intensity, cmap=cmap_RdW)  # , vmin=-im_max, vmax=im_max)
        cb = fig.colorbar(im)
        cb.set_label(r"Intensity (Normalized)")
        ax.set_xlabel(r"$x$ (px)")
        ax.set_ylabel(r"$y$ (px)")
        ax.set_xlim(-0.5, intensity.shape[1] - 0.5)
        ax.set_ylim(intensity.shape[0] - 0.5, -0.5)
        # plt.text(0, 2, r"P-V Phase: %0.3f $\mathrm{\mu m}$" % (im_max - im_min))
        # plt.text(
        #     0,
        #     4,
        #     r"RMS Phase: %0.3f $\mathrm{\mu m}$"
        #     % (np.std(phase, where=~np.isnan(phase))),
        # )
        return fig, ax, im, cb


class Zernike:
    rexp = {
        "sensor": re.compile(
            r"ASO Size : (?P<X>[0-9]{2}) x (?P<Y>[0-9]{2}); ASO Step : (?P<dx>.*) um x (?P<dy>.*) um"
        ),
        "pupil": re.compile(
            r"Pupil Center \(x, y\) : \((?P<pX>.*) mm x (?P<pY>.*) mm\) ; Pupil radius : (?P<pR>.*) mm;"
        ),
    }

    def __init__(self, filename):
        self.filename = filename
        data = np.loadtxt(self.filename, dtype="S", delimiter="\t", skiprows=6)
        # The P-V coefficients are simply dump multiplication factors on the polynomials
        self.coefficients = np.zeros(32)
        for i in range(32):
            self.coefficients[i] = float(data[i, 3])
        with open(self.filename, "r") as f:
            for line in f:
                key, match = parse_line(line, self.rexp)
                if key == "sensor":
                    self.width = int(match.group("X"))
                    self.height = int(match.group("Y"))
                    self.dx = float(match.group("dx"))
                    self.dy = float(match.group("dy"))
                if key == "pupil":
                    self.pX = float(match.group("pX"))
                    self.pY = float(match.group("pY"))
                    self.pR = float(match.group("pR"))
        self.poly_enum = {
            1: self.tilt0,
            2: self.tilt90,
            3: self.focus,
            4: self.astigmatism0,
            5: self.astigmatism45,
            6: self.coma0,
            7: self.coma90,
            8: self.spherical3rd,
            9: self.trefoil0,
            10: self.trefoil90,
            11: self.astigmatism5th0,
            12: self.astigmatism5th45,
            13: self.coma5th0,
            14: self.coma5th90,
            15: self.spherical5th,
            16: self.tetrafoil0,
            17: self.tetrafoil45,
            18: self.trefoil7th0,
            19: self.trefoil7th90,
            20: self.astigmatism7th0,
            21: self.astigmatism7th45,
            22: self.coma7th0,
            23: self.coma7th90,
            24: self.spherical7th,
            25: self.pentafoil0,
            26: self.pentafoil90,
            27: self.tetrafoil9th0,
            28: self.tetrafoil9th45,
            29: self.trefoil9th0,
            30: self.trefoil9th90,
            31: self.astigmatism9th0,
            32: self.astigmatism9th45,
        }
        self.names = np.array(
            [
                "Astigmatism0",
                "Astigmatism45",
                "Coma0",
                "Coma90",
                "Spherical3rd",
                "Trefoil0",
                "Trefoil90",
                "Astigmatism5th0",
                "Astigmatism5th45",
                "Coma5th0",
                "Coma5th90",
                "Spherical5th",
                "Tetrafoil0",
                "Tetrafoil45",
                "Trefoil7th0",
                "Trefoil7th90",
                "Astigmatism7th0",
                "Astigmatism7th45",
                "Coma7th0",
                "Coma7th90",
                "Spherical7th",
                "Pentafoil0",
                "Pentafoil90",
                "Tetrafoil9th0",
                "Tetrafoil9th45",
                "Trefoil9th0",
                "Trefoil9th90",
                "Astigmatism9th0",
                "Astigmatism9th45",
            ]
        )

    def polynomial_phase(self, ind, x, y):
        """Calculate the phase of a given polynomial on the passed grid. x and y should be in mm."""
        r = np.sqrt(x[None, :] ** 2 + y[:, None] ** 2)
        phi = np.arctan(y[:, None] / x[None, :])
        return self.poly_enum[ind](r, phi)

    def polynomial_phase_haso(self, ind):
        x = np.arange(self.width) * self.dx * 1e-3
        y = np.arange(self.height) * self.dy * 1e-3
        x -= 0.5 * x[-1] + self.pX
        y -= 0.5 * y[-1] - self.pY
        r = np.sqrt(x[None, :] ** 2 + y[:, None] ** 2)
        phi = -np.arctan(y[:, None] / x[None, :])
        sel = x < 0
        phi[:, sel] -= np.pi
        sel = r > self.pR
        r[sel] = np.NAN
        return self.poly_enum[ind](r, phi)

    def polynomial_sum_haso(self):
        """Sum all of the polynomials up on the grid for the haso."""
        phi = np.zeros((self.height, self.width))
        for i in range(len(self.coefficients)):
            phi += self.polynomial_phase_haso(i + 1)
        return phi

    def build_phase(self, inds, x, y):
        """Calculate the sum of polynomials on the passed grid. x and y should be in mm."""
        phase = np.zeros((len(y), len(x)))
        r = np.sqrt(x[None, :] ** 2 + y[:, None] ** 2)
        phi = np.arctan(y[:, None] / x[None, :])
        for ind in inds:
            phase += self.poly_enum[ind](r, phi)
        return phase

    def tilt0(self, r, phi):
        return self.coefficients[0] * (r / self.pR) * np.cos(phi)

    def tilt90(self, r, phi):
        return self.coefficients[1] * (r / self.pR) * np.sin(phi)

    def focus(self, r, phi):
        return self.coefficients[2] * (2 * (r / self.pR) ** 2 - 1)

    def astigmatism0(self, r, phi):
        return self.coefficients[3] * (r / self.pR) ** 2 * np.cos(2 * phi)

    def astigmatism45(self, r, phi):
        return self.coefficients[4] * (r / self.pR) ** 2 * np.sin(2 * phi)

    def coma0(self, r, phi):
        return (
            self.coefficients[5]
            * (3 * (r / self.pR) ** 2 - 2)
            * (r / self.pR)
            * np.cos(phi)
        )

    def coma90(self, r, phi):
        return (
            self.coefficients[6]
            * (3 * (r / self.pR) ** 2 - 2)
            * (r / self.pR)
            * np.sin(phi)
        )

    def spherical3rd(self, r, phi):
        return self.coefficients[7] * (
            6 * (r / self.pR) ** 4 - 6 * (r / self.pR) ** 2 + 1
        )

    def trefoil0(self, r, phi):
        return self.coefficients[8] * (r / self.pR) ** 3 * np.cos(3 * phi)

    def trefoil90(self, r, phi):
        return self.coefficients[9] * (r / self.pR) ** 3 * np.sin(3 * phi)

    def astigmatism5th0(self, r, phi):
        return (
            self.coefficients[10]
            * (4 * (r / self.pR) ** 2 - 3)
            * (r / self.pR) ** 2
            * np.cos(2 * phi)
        )

    def astigmatism5th45(self, r, phi):
        return (
            self.coefficients[11]
            * (4 * (r / self.pR) ** 2 - 3)
            * (r / self.pR) ** 2
            * np.sin(2 * phi)
        )

    def coma5th0(self, r, phi):
        return (
            self.coefficients[12]
            * (10 * (r / self.pR) ** 4 - 12 * (r / self.pR) ** 2 + 3)
            * (r / self.pR)
            * np.cos(phi)
        )

    def coma5th90(self, r, phi):
        return (
            self.coefficients[13]
            * (10 * (r / self.pR) ** 4 - 12 * (r / self.pR) ** 2 + 3)
            * (r / self.pR)
            * np.sin(phi)
        )

    def spherical5th(self, r, phi):
        return self.coefficients[14] * (
            20 * (r / self.pR) ** 6
            - 30 * (r / self.pR) ** 4
            + 12 * (r / self.pR) ** 2
            - 1
        )

    def tetrafoil0(self, r, phi):
        return self.coefficients[15] * (r / self.pR) ** 4 * np.cos(4 * phi)

    def tetrafoil45(self, r, phi):
        return self.coefficients[16] * (r / self.pR) ** 4 * np.sin(4 * phi)

    def trefoil7th0(self, r, phi):
        return (
            self.coefficients[17]
            * (5 * (r / self.pR) ** 2 - 4)
            * (r / self.pR) ** 3
            * np.cos(3 * phi)
        )

    def trefoil7th90(self, r, phi):
        return (
            self.coefficients[18]
            * (5 * (r / self.pR) ** 2 - 4)
            * (r / self.pR) ** 3
            * np.sin(3 * phi)
        )

    def astigmatism7th0(self, r, phi):
        return (
            self.coefficients[19]
            * (15 * (r / self.pR) ** 4 - 20 * (r / self.pR) ** 2 + 6)
            * (r / self.pR) ** 2
            * np.cos(2 * phi)
        )

    def astigmatism7th45(self, r, phi):
        return (
            self.coefficients[20]
            * (15 * (r / self.pR) ** 4 - 20 * (r / self.pR) ** 2 + 6)
            * (r / self.pR) ** 2
            * np.sin(2 * phi)
        )

    def coma7th0(self, r, phi):
        return (
            self.coefficients[21]
            * (
                35 * (r / self.pR) ** 6
                - 60 * (r / self.pR) ** 4
                + 30 * (r / self.pR) ** 2
                - 4
            )
            * (r / self.pR)
            * np.cos(phi)
        )

    def coma7th90(self, r, phi):
        return (
            self.coefficients[22]
            * (
                35 * (r / self.pR) ** 6
                - 60 * (r / self.pR) ** 4
                + 30 * (r / self.pR) ** 2
                - 4
            )
            * (r / self.pR)
            * np.sin(phi)
        )

    def spherical7th(self, r, phi):
        return self.coefficients[23] * (
            70 * (r / self.pR) ** 8
            - 140 * (r / self.pR) ** 6
            + 90 * (r / self.pR) ** 4
            - 20 * (r / self.pR) ** 2
            + 1
        )

    def pentafoil0(self, r, phi):
        return self.coefficients[24] * (r / self.pR) ** 5 * np.cos(5 * phi)

    def pentafoil90(self, r, phi):
        return self.coefficients[25] * (r / self.pR) ** 5 * np.cos(5 * phi)

    def tetrafoil9th0(self, r, phi):
        return (
            self.coefficients[26]
            * (6 * (r / self.pR) ** 2 - 5)
            * (r / self.pR) ** 4
            * np.cos(4 * phi)
        )

    def tetrafoil9th45(self, r, phi):
        return (
            self.coefficients[27]
            * (6 * (r / self.pR) ** 2 - 5)
            * (r / self.pR) ** 4
            * np.sin(4 * phi)
        )

    def trefoil9th0(self, r, phi):
        return (
            self.coefficients[28]
            * (21 * (r / self.pR) ** 4 - 30 * (r / self.pR) ** 2 + 10)
            * (r / self.pR) ** 3
            * np.cos(3 * phi)
        )

    def trefoil9th90(self, r, phi):
        return (
            self.coefficients[29]
            * (21 * (r / self.pR) ** 4 - 30 * (r / self.pR) ** 2 + 10)
            * (r / self.pR) ** 3
            * np.sin(3 * phi)
        )

    def astigmatism9th0(self, r, phi):
        return (
            self.coefficients[30]
            * (
                56 * (r / self.pR) ** 6
                - 105 * (r / self.pR) ** 4
                + 60 * (r / self.pR) ** 2
                - 10
            )
            * (r / self.pR) ** 2
            * np.cos(2 * phi)
        )

    def astigmatism9th45(self, r, phi):
        return (
            self.coefficients[31]
            * (
                56 * (r / self.pR) ** 6
                - 105 * (r / self.pR) ** 4
                + 60 * (r / self.pR) ** 2
                - 10
            )
            * (r / self.pR) ** 2
            * np.sin(2 * phi)
        )

    # Visualization functions -------------------------------------------------
    # --------------------------------------------------------------------------
    def plot_polynomials(self):
        fig = plt.figure(figsize=(5.4, 3), dpi=150)
        ax = fig.add_subplot(111)
        colors = ["tab:blue" if m > 0 else "tab:red" for m in self.coefficients[3:]]
        ax.bar(self.names, abs(self.coefficients[3:]), zorder=3, color=colors)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_ylabel(r"$\left|C_i\right|$ ($\mathrm{\mu m}$)")
        ax.grid(axis="y", zorder=0)
        ax.text(
            0.98,
            0.95,
            r"Pupil Radius: %0.3f $\mathrm{mm}$" % (self.pR),
            ha="right",
            va="top",
            transform=ax.transAxes,
        )
        return fig, ax
