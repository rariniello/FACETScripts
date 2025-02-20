import os
import datetime
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

import load
import image


class DATASET:
    def __init__(self, experiment: str, number: str):
        self.experiment = experiment
        self.number = number
        self.dataPath = load.dataPath
        self.experimentPath = os.path.join(self.dataPath, experiment)
        self.datasetPath = os.path.join(
            self.dataPath, experiment, "{}_{}".format(experiment, number)
        )
        self.dataStructPath = os.path.join(
            self.datasetPath, "{}_{}.mat".format(experiment, number)
        )

        self.loadDataStruct()

    def loadDataStruct(self):
        # Only the 'data_struct' key contains useful info
        self._data = scipy.io.loadmat(self.dataStructPath, simplify_cells=True)[
            "data_struct"
        ]
        # Everything entered into the DAQ window when setting up the dataset
        self.params = self._data["params"]
        self.cameras = self.params["camNames"]
        if isinstance(self.cameras, str):
            self.cameras = [self.cameras]
        self.scan_vals = self.params["scanVals"]
        self.Nsteps = len(self.scan_vals)
        self.HDF5 = False
        if "saveMethod" in self.params:
            if self.params["saveMethod"] == "HDF5":
                self.HDF5 = True
        # Metadata for each camera and all the PVs in scalar lists
        self.metadata = self._data["metadata"]
        self.loadCameraCalibration()
        # Timestamps and nas path info
        self.saveInfo = self._data["save_info"]
        self.timestamp = self.arrayToDatetime(self.saveInfo["local_time"])
        # Pulse IDs used for matching shots
        self.pulseID = self._data["pulseID"]
        # Path and filename of every image in the dataset
        self.images = self._data["images"]
        # Scalar data and common indexes
        self.common_index = self._data["scalars"]["common_index"]
        self.N = len(self.common_index)
        self.common_stepIndex = self._data["scalars"]["steps"][self.common_index - 1]
        if len(self.scan_vals) == 0:
            self.x = None
        else:
            self.x = self.scan_vals[self.common_stepIndex - 1]
        # All the steps that have matches
        steps = self.common_stepIndex
        unique_steps = np.unique(steps)
        if self.Nsteps > 0:
            self.x_steps = self.scan_vals[unique_steps - 1]
        else:
            self.x_steps = None

    def loadCameraCalibration(self):
        for cam in self.cameras:
            bin = self.metadata[cam]["BinX_RBV"]
            cal = self.metadata[cam]["RESOLUTION"] * bin
            if np.isnan(cal):
                return
            image.set_calibration(cam, cal * 1e-3)

    # Data retrieval functions -------------------------------------------------
    # --------------------------------------------------------------------------
    def getScalar(self, list: str, PV: str) -> np.ndarray:
        PV = PV.replace(":", "_")
        PV = PV.replace(".", "_")
        try:
            return self._data["scalars"][list][PV][self.common_index - 1]
        except IndexError:
            # The scalar array probably has too few shots
            a = np.empty(self.N)
            a[:] = np.nan
            data = self._data["scalars"][list][PV]
            sel = (self.common_index - 1) < len(data)
            a[sel] = data[self.common_index[sel] - 1]
            return a

    def getRawScalar(self, list: str, PV: str) -> np.ndarray:
        PV = PV.replace(":", "_")
        return self._data["scalars"][list][PV]

    def getImage(self, camera, ind):
        step = self.common_stepIndex[ind]
        common_index = self.pulseID[camera + "common_index"][ind]
        if self.HDF5:
            image_path = self.images[camera]["loc"]
            filename = os.path.basename(image_path)
            ind = common_index - 1 - (step - 1) * self.params["n_shot"]
            return image.HDF5_DAQ(camera, self, filename, ind, step)
        else:
            image_path = self.images[camera]["loc"][common_index - 1]
            filename = os.path.basename(image_path)
            ind = int(image_path[-8:-4])
            return image.DAQ(camera, self, filename, ind, step)

    def getImage_NoMatch(self, camera, ind):
        image_path = self.images[camera]["loc"][ind]
        filename = os.path.basename(image_path)
        step = self.images[camera]["step"][ind]
        return image.DAQ(camera, self, filename, ind, step)

    def getImageBackground(self, camera):
        XOrient = self.metadata[camera]["X_ORIENT"]
        YOrient = self.metadata[camera]["Y_ORIENT"]
        if "IS_ROTATED" in self.metadata[camera]:
            isRotated = self.metadata[camera]["IS_ROTATED"]
        else:
            isRotated = None
        bkgd = self._data["backgrounds"][camera]
        if bkgd.ndim == 3:
            bkgd = np.average(bkgd, axis=2)
            bkgd = image.orientImage(np.transpose(bkgd), XOrient, YOrient, isRotated)
            bkgd = image.specialFlips(camera, bkgd, isRotated)
            return bkgd
        elif bkgd.ndim == 2:
            bkgd = image.orientImage(np.transpose(bkgd), XOrient, YOrient, isRotated)
            bkgd = image.specialFlips(camera, bkgd, isRotated)
            return bkgd

    def scalarLists(self) -> list:
        scalarLists = []
        # If there is only one list it is a string and needs to be converted to an array
        for item in np.array(self.params["BSA_list"], ndmin=1):
            scalarLists.append(item)
        for item in np.array(self.params["nonBSA_list"], ndmin=1):
            scalarLists.append(item)
        return scalarLists

    def PVsInList(self, list: str) -> np.ndarray:
        return self.metadata[list]["PVs"]

    # Util functions -----------------------------------------------------------
    # --------------------------------------------------------------------------
    def getListForPV(self, PV):
        lists = self.scalarLists()
        for l in lists:
            pvs = self.PVsInList(l)
            if PV in pvs:
                return l
        return None

    def arrayToDatetime(self, a: np.ndarray) -> datetime.datetime:
        year = int(a[0])
        month = int(a[1])
        day = int(a[2])
        hour = int(a[3])
        minute = int(a[4])
        second = int(math.floor(a[5]))
        microsecond = int((a[5] % 1) * 1000000)
        return datetime.datetime(year, month, day, hour, minute, second, microsecond)

    # Analysis functions -------------------------------------------------------
    # --------------------------------------------------------------------------
    def averageByStep(
        self, data, outlierRejection=False, returnOutliers=False, sel=None
    ):
        """Divides the given scalar values into scan steps and averages within each step.

        Values beyond 1.5 times the interquartile range can be rejected by setting rejectOutlier
        to True. Similarly, only certain data can be included by passing a boolean array of length
        self.N to use as a selector on the data (False will be ignored)

        Args:
            data: Scalar values for each common index, must have length self.N.
            outlierRejection: Whether to reject outliers from the average and standard deviation.
            returnOutliers: Whether to return the indices of the outlier data or not.
            sel: Optional selector array used to select which data to keep.

        Returns:
            averages: Average value of the data within each step.
            std: Standard deviation of the data within each step.
            stdMeans: Standard deviation of the mean of the dat within each step.
            outliers: Only returned in returnOutlier=True, indexes of outliers that were rejected.
        """
        steps = self.common_stepIndex
        unique_steps = np.unique(steps)

        num_steps = len(unique_steps)
        averages = np.zeros(num_steps)
        stds = np.zeros(num_steps)
        stdMeans = np.zeros(num_steps)

        outlier = 0
        if sel is None:
            sel = np.full(self.N, True)

        if outlierRejection:
            index = np.arange(self.N)
            outliers = np.array([], dtype="int")
            for i, step in enumerate(unique_steps):
                stepSel = (steps == step) * sel
                step_data = data[stepSel]

                # Calculate interquartile range
                q1, q3 = np.percentile(step_data, [25, 75])
                iqr = q3 - q1

                # Define the lower and upper bounds for outliers
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                # Filter out outliers
                keepSel = (step_data >= lower_bound) & (step_data <= upper_bound)
                filtered_data = step_data[keepSel]
                outliers = np.append(outliers, index[stepSel][~keepSel])
                outlier += len(step_data) - len(filtered_data)

                # Calculate statistics for the filtered data
                averages[i] = np.mean(filtered_data)
                stds[i] = np.std(filtered_data)
                stdMeans[i] = np.std(filtered_data) / np.sqrt(len(filtered_data))
            print(f"{outlier} outliers removed from the data.")
            if returnOutliers:
                return averages, stds, stdMeans, outliers
            else:
                return averages, stds, stdMeans
        else:
            for i, step in enumerate(unique_steps):
                stepSel = (steps == step) * sel
                step_data = data[stepSel]

                # Calculate statistics for the data
                averages[i] = np.mean(step_data)
                stds[i] = np.std(step_data)
                stdMeans[i] = np.std(step_data) / np.sqrt(len(step_data))

            return averages, stds, stdMeans

    # Visualization functions --------------------------------------------------
    # --------------------------------------------------------------------------
    def blankFigure(self):
        fig = plt.figure(figsize=(4, 3), dpi=300)
        ax = fig.add_subplot()
        ax.dataset_text = ax.text(
            0.01,
            0.95,
            "{}, Dataset: {}".format(self.experiment, self.number),
            transform=ax.transAxes,
        )
        return fig, ax

    def plotRawByStep(self, pv=None, scalar=None):
        if pv is not None:
            l = self.getListForPV(pv)
            data = self.getScalar(l, pv)
        elif scalar is not None:
            data = scalar
        else:
            return None
        fig, ax = self.blankFigure()
        ax.plot(self.x, data, ".", markersize=3)
        ax.set_xlabel(self.params["scanPVs"])
        if pv is not None:
            ax.set_ylabel(pv)
        return fig, ax

    def plotScalarByStep(self, pv=None, scalar=None, outlierRejection=False, sel=None):
        x = self.x_steps
        if pv is not None:
            l = self.getListForPV(pv)
            data = self.getScalar(l, pv)
        elif scalar is not None:
            data = scalar
        else:
            return None
        y, y_stdy, y_stdMean = self.averageByStep(
            data, outlierRejection=outlierRejection, sel=sel
        )
        fig, ax = self.blankFigure()
        ax.errorbar(x, y, yerr=y_stdy, fmt=".", markersize=3)
        ax.set_xlabel(self.params["scanPVs"])
        if pv is not None:
            ax.set_ylabel(pv)
        return fig, ax

    def plotCorrelation(self, pvs=None, scalars=None, fit=False, sel=None):
        if pvs is not None and len(pvs) == 2:
            l = self.getListForPV(pvs[0])
            x_data = self.getScalar(l, pvs[0])
            l = self.getListForPV(pvs[1])
            y_data = self.getScalar(l, pvs[1])
        elif scalars is not None and len(scalars) == 2:
            x_data = scalars[0]
            y_data = scalars[1]
        if sel is None:
            N = len(x_data)
            sel = np.full(N, True)
        corrcoef = np.corrcoef(x_data[sel], y_data[sel])
        fig, ax = self.blankFigure()
        ax.plot(x_data[sel], y_data[sel], ".", markersize=2)
        if pvs is not None:
            ax.set_xlabel(pvs[0])
            ax.set_ylabel(pvs[1])
        ax.text(
            0.99,
            0.95,
            "Correlation coefficient {:0.3f}".format(corrcoef[0, 1]),
            transform=ax.transAxes,
            ha="right",
        )
        if fit:
            z = np.polyfit(x_data[sel], y_data[sel], 1)
            p = np.poly1d(z)
            x = [np.min(x_data[sel]), np.max(x_data[sel])]
            ax.plot(x, p(x), c="tab:blue")
            ax.text(
                0.99,
                0.05,
                r"Fit: $y=$"
                + "{:0.3e}".format(z[0])
                + r"$x+$"
                + "{:0.3e}".format(z[1]),
                transform=ax.transAxes,
                ha="right",
                va="top",
            )
            return fig, ax, z
        return fig, ax

    def correlationGrid(self, pvs=[], scalars=[]):
        N = len(pvs) + len(scalars)
        data = np.zeros((N, self.N))
        for i in range(len(pvs)):
            l = self.getListForPV(pvs[i])
            data[i] = self.getScalar(l, pvs[i])
        for i in range(len(pvs), N):
            data[i] = scalars[i - len(pvs)]
        cor = np.corrcoef(data)
        fig = plt.figure(figsize=(1.2 * N, 1.2 * N), dpi=150)
        gs = gridspec.GridSpec(N, N, wspace=0, hspace=0)
        axa = {}

        colors = plt.cm.Spectral(np.linspace(0, 1, 101))
        for i in range(N):
            for j in range(i, N):
                ax_name = "{:02d}{:02d}".format(i, j)
                # Below the top row and right of the forst plot should share both axes with previous plot
                if i > 0 and j > i:
                    ax = axa[ax_name] = fig.add_subplot(
                        gs[i, j],
                        sharex=axa["{:02d}{:02d}".format(0, j)],
                        sharey=axa["{:02d}{:02d}".format(i, i)],
                    )
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                # Plots in the top row should only share y
                elif i == 0 and j > i:
                    ax = axa[ax_name] = fig.add_subplot(
                        gs[i, j], sharey=axa["{:02d}{:02d}".format(i, i)]
                    )
                    plt.setp(ax.get_yticklabels(), visible=False)
                # Plots in the front of each row should only share x
                elif j == i and i != 0:
                    ax = axa[ax_name] = fig.add_subplot(
                        gs[i, j], sharex=axa["{:02d}{:02d}".format(0, j)]
                    )
                else:
                    ax = axa[ax_name] = fig.add_subplot(gs[i, j])

                # Do the actual plotting
                if i == 0:
                    ax.set_title(pvs[j], fontsize=6)
                if j == N - 1:
                    ax.set_ylabel(pvs[i], fontsize=6)
                    ax.yaxis.set_label_position("right")
                if np.isnan(cor[i, j]):
                    continue
                c = colors[int(50 * (cor[i, j] + 1))]
                ax.plot(data[j, :], data[i, :], ".", markersize=2, c=c)
                ax.annotate(
                    r"Corr={:0.3f}".format(cor[i, j]),
                    xy=(0, 1),
                    xytext=(3, -3),
                    xycoords="axes fraction",
                    textcoords="offset points",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6,
                )
        fig.suptitle("{}, Dataset: {}".format(self.experiment, self.number))
        return fig, axa

    def outlierComparisonPlot(self, pv=None, scalar=None):
        x = self.scan_vals
        if pv is not None:
            l = self.getListForPV(pv)
            data = self.getScalar(l, pv)
        elif scalar is not None:
            data = scalar
        else:
            return None
        y, y_stdy, y_stdMean = self.averageByStep(data, outlierRejection=False)
        y_out, y_stdy_out, y_stdMean_out = self.averageByStep(
            data, outlierRejection=True
        )
        fig, ax = self.blankFigure()
        ax.errorbar(x, y, yerr=y_stdy, fmt=".", markersize=3, label="Raw data")
        ax.errorbar(
            x, y_out, yerr=y_stdy_out, fmt=".", markersize=3, label="Outliers removed"
        )
        ax.set_xlabel(self.params["scanPVs"])
        if pv is not None:
            ax.set_ylabel(pv)
        ax.legend()
        return fig, ax

    def cameraGif(self, cam, name, M=1, cal=True, offset=0, clim=None):
        """Makes a gif showing all of the shots from a given camera"""
        # TODO add a keyword argument to turn background subtraction on/off
        N = len(self.common_index)
        im = self.getImage(cam, 0 + offset)
        bkgd = self.getImageBackground(cam)
        im.subtract_background(bkgd)
        fig, ax, im, cb, ext = im.plot_image(cal=cal)
        if clim is not None:
            im.set_clim(clim)
        plt.tight_layout()

        def animate(i):
            img = self.getImage(cam, i * M + offset)
            img.subtract_background(bkgd)
            data = img.data
            im.set_data(data)
            ax.stepText.set_text("Shot: {:04d}\nStep: {:02d}".format(img.ind, img.step))
            return im

        ani = animation.FuncAnimation(
            fig, animate, repeat=True, frames=int(N / M), interval=50
        )
        writer = animation.PillowWriter(fps=15, bitrate=1800)
        ani.save(name, writer=writer)
        print("Finished creating GIF")
