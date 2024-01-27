import math

import numpy as np
import matplotlib.pyplot as plt

from image import IMAGE, Elog, DAQ
import image
import plot


def forward(y, d_nom, E_bend, dy):
    return d_nom * E_bend / (dy - y)


def inverse(E, d_nom, E_bend, dy):
    return dy - d_nom * E_bend / E


class LFOV:
    def __init__(self, image):
        self.image = image
        self.d_nom = 0.0
        self.E_bend = 0.0
        self.dy = 0.0

    def forward(self, y):
        return forward(y, self.d_nom, self.E_bend, self.dy)

    def inverse(self, y):
        return inverse(y, self.d_nom, self.E_bend, self.dy)

    def plot_image(self, cmap=plot.roar, metadata=True):
        fig, ax, im, cb, ext = self.image.plot_image(
            cal=True, cmap=cmap, metadata=metadata
        )
        E_ticks = np.arange(
            math.ceil(self.forward(ext[3])), math.floor(self.forward(ext[2]))
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
        return fig, ax, im, cb, ext


class LFOV_DS:
    def __init__(self, dataset):
        self.dataset = dataset
        self.N = len(dataset.common_index)
        self.img = dataset.getImage("LFOV", 0)

    def forward(self, y):
        return forward(y, self.d_nom, self.E_bend, self.dy)

    def inverse(self, y):
        return inverse(y, self.d_nom, self.E_bend, self.dy)

    def get_waterfall(self):
        cam = "LFOV"
        ds = self.dataset
        N = self.N
        img = ds.getImage(cam, 0)
        waterfall = np.zeros((N, img.height))
        bkgd = ds.getImageBackground(cam)
        for i in range(N):
            im = ds.getImage(cam, i)
            im.subtract_background(bkgd)
            proj = np.sum(im.data, axis=1)
            waterfall[i, :] = proj
        self.waterfall = waterfall
        return waterfall

    def plot_waterfall(self):
        h = self.img.height
        cal = image.cal["LFOV"]
        ext = [-0.5, self.N + 0.5, -0.5 * cal, (h + 0.5) * cal]
        fig = plt.figure(figsize=(4.85, 5), dpi=300)
        ax = plt.subplot()
        im = ax.imshow(
            np.transpose(self.waterfall),
            cmap=plot.roar,
            aspect="auto",
            extent=ext,
        )
        ax.set_xlabel("Shot number")
        ax.set_ylabel("Energy (GeV)")
        cb = fig.colorbar(im)
        cb.set_label("Counts")
        # Shifts the scale to be linear in energy
        # plt.yscale('function', functions=(forward, inverse))

        E_ticks = np.arange(
            math.ceil(self.forward(ext[2])), math.floor(self.forward(ext[3]))
        )
        tick_locs = [self.inverse(E) for E in E_ticks]
        tick_lbls = ["{:0.1f}".format(E) for E in E_ticks]
        ax.set_yticks(tick_locs, tick_lbls)
        datasetText = ax.annotate(
            "{}, Dataset: {}\n{}".format(
                self.dataset.experiment, self.dataset.number, "LFOV"
            ),
            xy=(0, 1),
            xytext=(3, -3),
            xycoords="axes fraction",
            textcoords="offset points",
            color="k",
            va="top",
        )
        return fig, ax, im, cb, ext
