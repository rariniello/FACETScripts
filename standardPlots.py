import os
import datetime
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt


def plotGaussianFit(
    y,
    ds,
):
    x = ds.scan_vals
    y, y_stdy, y_stdMean = ds.averageByStep(sumCounts)
    fg, popt = an.fit_gaussian(x, y, p0=(26.6, 2.5e8, 0.2, 1e8), sigma=y_stdy)
    xf = np.linspace(x[0], x[-1], 1000)
    fig, ax = ds.plotScalarByStep(scalar=sumCounts)
    ax.set_xlabel("Grating motor position (mm)")
    ax.set_ylabel("Total counts on {}".format(cam))
    ax.plot(xf, fg(xf, *px), c="tab:blue")
