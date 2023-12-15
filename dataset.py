import os
import datetime
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt

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

    def loadCameraCalibration(self):
        for cam in self.cameras:
            cal = self.metadata[cam]["RESOLUTION"]
            image.set_calibration(cam, cal * 1e-3)

    # Data retrieval functions -------------------------------------------------
    # --------------------------------------------------------------------------
    def getScalar(self, list: str, PV: str) -> np.ndarray:
        PV = PV.replace(":", "_")
        return self._data["scalars"][list][PV][self.common_index]

    def getImage(self, camera, ind):
        common_index = self.pulseID[camera + "common_index"][ind]
        image_path = self.images[camera]["loc"][common_index - 1]
        filename = os.path.basename(image_path)
        step = self.common_stepIndex[ind]
        ind = int(image_path[-8:-4])
        return image.DAQ(camera, self, filename, ind, step)

    def getImage_NoMatch(self, camera, ind):
        image_path = self.images[camera]["loc"][ind]
        filename = os.path.basename(image_path)
        step = self.images[camera]["step"][ind]
        return image.DAQ(camera, self, filename, ind, step)

    def getImageBackground(self, camera):
        bkgd = self._data["backgrounds"][camera]
        if bkgd.ndim == 3:
            return np.transpose(np.average(bkgd, axis=2))
        elif bkgd.ndim == 2:
            return np.transpose(bkgd)

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
            if PV in l:
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
    def averageByStep(self, data, outlierRejection=False):
        steps = self.common_stepIndex
        unique_steps = np.unique(steps)

        num_steps = len(unique_steps)
        averages = np.zeros(num_steps)
        stds = np.zeros(num_steps)
        stdMeans = np.zeros(num_steps)

        outlier = 0

        if outlierRejection:
            for i, step in enumerate(unique_steps):
                step_data = data[steps == step]

                # Calculate interquartile range
                q1, q3 = np.percentile(step_data, [25, 75])
                iqr = q3 - q1

                # Define the lower and upper bounds for outliers
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                # Filter out outliers
                filtered_data = step_data[
                    (step_data >= lower_bound) & (step_data <= upper_bound)
                ]
                outlier += len(step_data) - len(filtered_data)

                # Calculate statistics for the filtered data
                averages[i] = np.mean(filtered_data)
                stds[i] = np.std(filtered_data)
                stdMeans[i] = np.std(filtered_data) / np.sqrt(len(filtered_data))
            print(f"{outlier} outliers removed from the data.")

        else:
            for i, step in enumerate(unique_steps):
                step_data = data[steps == step]

                # Calculate statistics for the data
                averages[i] = np.mean(step_data)
                stds[i] = np.std(step_data)
                stdMeans[i] = np.std(step_data) / np.sqrt(len(step_data))

        return averages, stds, stdMeans

    # Visualization functions --------------------------------------------------
    # --------------------------------------------------------------------------
    def plotRawByStep(self, pv=None, scalar=None):
        if pv is not None:
            l = self.getListForPV(pv)
            data = self.getScalar(l, pv)
        elif scalar is not None:
            data = scalar
        else:
            return None
        fig = plt.figure(figsize=(4, 3), dpi=300)
        ax = fig.add_subplot()
        ax.plot(self.x, data, ".", markersize=3)
        ax.set_xlabel(self.params["scanPVs"])
        if pv is not None:
            ax.set_ylabel(pv)
        ax.text(
            0.01,
            0.95,
            "{}, Dataset: {}".format(self.experiment, self.number),
            transform=ax.transAxes,
        )
        return fig, ax

    def plotScalarByStep(self, pv=None, scalar=None, outlierRejection=False):
        x = self.scan_vals
        if pv is not None:
            l = self.getListForPV(pv)
            data = self.getScalar(l, pv)
        elif scalar is not None:
            data = scalar
        else:
            return None
        y, y_stdy, y_stdMean = self.averageByStep(data, outlierRejection)
        fig = plt.figure(figsize=(4, 3), dpi=300)
        ax = fig.add_subplot()
        ax.errorbar(x, y, yerr=y_stdy, fmt=".", markersize=3)
        ax.set_xlabel(self.params["scanPVs"])
        if pv is not None:
            ax.set_ylabel(pv)
        ax.text(
            0.01,
            0.95,
            "{}, Dataset: {}".format(self.experiment, self.number),
            transform=ax.transAxes,
        )
        return fig, ax

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
        fig = plt.figure(figsize=(4, 3), dpi=300)
        ax = fig.add_subplot()
        ax.errorbar(x, y, yerr=y_stdy, fmt=".", markersize=3, label="Raw data")
        ax.errorbar(
            x, y_out, yerr=y_stdy_out, fmt=".", markersize=3, label="Outliers removed"
        )
        ax.set_xlabel(self.params["scanPVs"])
        if pv is not None:
            ax.set_ylabel(pv)
        ax.text(
            0.01,
            0.95,
            "{}, Dataset: {}".format(self.experiment, self.number),
            transform=ax.transAxes,
        )
        ax.legend()
        return fig, ax
