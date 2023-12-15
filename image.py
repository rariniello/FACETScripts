import numpy as np
from PIL import Image
from scipy import ndimage
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.transforms as mtransforms
from scipy.optimize import curve_fit
import scipy.io
import os
import load


cal = {}


def set_calibration(cam_name: str, calibration: float):
    """Adds/updates the calibration list entry for the given camera.

    Args:
        cam_name: Name of the camera, must match the name in the metadata.
        calibration: Pixel calibration in mm/px.
    """
    cal[str(cam_name)] = calibration


class IMAGE:
    # Initialization functions ------------------------------------------------
    # --------------------------------------------------------------------------
    def __init__(self, camera, **kwargs):
        self.camera = str(camera)
        self.image = self.load_image()
        self.meta = self.get_image_meta()
        self.data = np.array(self.image, dtype="float")
        if self.camera not in cal:
            set_calibration(self.camera, 1)
            print(
                "Warning, calibration was not defined for camera {}, defaulting to 1.".format(
                    self.camera
                )
            )
        self.cal = cal[self.camera]  # In mm/px
        self.width = self.image.width
        self.height = self.image.height
        self.xp = np.arange(0, self.width, 1)
        self.yp = np.arange(0, self.height, 1)
        self.x = self.cal * self.xp
        self.y = self.cal * self.yp
        self.center = None
        self.check_image()

    def load_image(self) -> object:
        """Loads an image from a given data set.

        Overwrite to define how to load an image.

        Returns:
            image: Pillow image object for the tiff.
        """
        image = Image.new("I", (1292, 964))
        return image

    def get_image_meta(self):
        """Return the meta data dictionary from a pillow image object.

        Overwrite to define how to load the metadata for an image.

        Returns
        -------
        meta : dict
            The meta data dictionary contained in the tiff image.
        """
        meta = {}
        return meta

    def check_image(self):
        """Verify meta data in the tiff is consistent with filename/image."""
        # if "Width" in self.meta and self.meta["Width"] != self.width:
        #     print(
        #         "Image meta data width {:d} does not match image width {:d}".format(
        #             self.meta["Width"], self.width
        #         )
        #     )
        # if "Height" in self.meta and self.meta["Height"] != self.height:
        #     print(
        #         "Image meta data height {:d} does not match image height {:d}".format(
        #             self.meta["Height"], self.height
        #         )
        #     )

    def refresh_calibration(self):
        """Update the camera calibration if it has been changed."""
        self.cal = cal[self.camera]
        self.x = self.cal * self.xp
        self.y = self.cal * self.yp

    # Modification functions --------------------------------------------------
    # --------------------------------------------------------------------------
    def rotate(self, angle: float):
        """Rotate the image.

        Args:
            angle: Angle to rotate the image by, in deg.
        """
        self.image = self.image.rotate(angle)
        self.data = np.array(self.image, dtype="float")

    def crop(self, box: tuple):
        """Crops the image.

        Args:
            box: The crop rectangle, as a (left, upper, right, lower) tuple.
        """
        self.image = self.image.crop(box)
        self.data = np.array(self.image, dtype="float")
        self.width = self.image.width
        self.height = self.image.height
        self.xp = np.arange(0, self.width, 1)
        self.yp = np.arange(0, self.height, 1)
        self.x = self.cal * self.xp
        self.y = self.cal * self.yp
        self.center = None

    def center_image(self, strategy: str, o: int, **kwargs):
        """Center the image by non-uniformaly padding it. Meta will no longer match class parameters.

        Args:
            strategy: See calculate_center for available strategies and **kwargs.
            o: Padding on each side of returned array, in px.
        """
        cen_image, center = self.get_center_image(strategy, o, **kwargs)
        self.data = cen_image
        self.image = Image.fromarray(cen_image)
        self.width = self.image.width
        self.height = self.image.height
        self.center = (self.width / 2, self.height / 2)
        self.xp = np.arange(0, self.width, 1)
        self.yp = np.arange(0, self.height, 1)
        self.x = self.cal * self.xp
        self.y = self.cal * self.yp

    def subtract_background_noise(self, threshold):
        data = self.data
        sel = data < threshold
        background = data[sel]
        avg_background = np.average(background)
        self.data = data - avg_background
        self.image = Image.fromarray(self.data)

    def subtract_background(self, background):
        data = self.data
        self.data = data - background
        self.image = Image.fromarray(self.data)

    # Calculation functions ---------------------------------------------------
    # --------------------------------------------------------------------------
    def calculate_center(
        self,
        strategy: str = "cm",
        threshold: int = 12,
        f=None,
        p0: tuple = None,
        center: np.ndarray = None,
    ):
        """Calculate the pixel location of the center of the image.

        Args:
            strategy: Select the technique to use to find the center of the image.
                'cm' - the image center is found by taking the cm of the image.
                'mask' - a mask is formed from all pixels with values greather than a threshold.
                    The image center is the centroid of the mask.
                'fit' - fit a function to the image to find the center.
                'external' - pass in the location of the mask center.
            threshold: When strategy='mask', threshold is used to create the mask.
            f: When strategy='f', function to fit to the data. The first two free parameters
                should be the x and y positions of the center. It should accept as the first
                arguments (x, y).
            p0: When strategy='f', initial guesses for model parameters.
            center: When strategy='external', location of the image center.
        """
        if strategy == "cm":
            self.center = self.center_cm()
        if strategy == "mask":
            self.center = self.center_mask(threshold)
        if strategy == "fit":
            self.fit = self.center_fit(f, p0)
            self.f = f
            self.center = np.array([self.fit[0], self.fit[1]])
        if strategy == "external":
            self.center = center

    def center_cm(self):
        """Calculate the center of mass of the image."""
        return np.flip(ndimage.center_of_mass(self.data))

    def center_mask(self, threshold):
        """Calculate the centroid of a mask of the image."""
        mask = self.data > threshold
        return np.flip(ndimage.center_of_mass(mask))

    def center_fit(self, f, p0):
        """Fit a function to the data to find the center."""
        X, Y = np.meshgrid(self.xp, self.yp)
        Z = self.data
        xdata = np.vstack((X.ravel(), Y.ravel()))
        ydata = Z.ravel()
        popt, pcov = curve_fit(f, xdata, ydata, p0=p0)
        return popt

    def calculate_energy(self):
        """Calculate the total sum of all the pixels."""
        return np.sum(self.data)

    def get_center_image(self, strategy, o, **kwargs):
        """Return a version of the image that is centered.

        Parameters
        ----------
        strategy : string
            See calculate_center for available strategies and **kwargs.
        o : int
            Padding on each side of returned array, in px.

        Returns
        -------
        cen_image : array of floats
            Padded image with the actual image shifted to be centered.
        center : tuple of floats
            The location of the image center in pixel coordinates.
        """
        self.calculate_center(strategy, **kwargs)
        cen = np.array([self.width / 2 + o, self.height / 2 + o], dtype="int")
        center = self.center
        if center is None:
            center = (self.width / 2, self.height / 2)
        shift = np.array(np.rint(cen - center), dtype="int")
        cen_image = np.zeros((self.height + 2 * o, self.width + 2 * o))
        start_y = shift[1]
        end_y = start_y + self.height
        start_x = shift[0]
        end_x = start_x + self.width
        cen_image[start_y:end_y, start_x:end_x] = self.data
        return cen_image, center

    # Visualization functions --------------------------------------------------
    # --------------------------------------------------------------------------
    def get_ext(self, cal=True):
        """Helper function to get the extent for imshow."""
        if cal:
            ext = self.cal * np.array([-0.5, self.width + 0.5, self.height + 0.5, -0.5])
        else:
            ext = np.array([-0.5, self.width + 0.5, self.height + 0.5, -0.5])
        return ext

    def create_fig_ax(self, cal=True):
        """Create the figure and ax objects for plotting the image.

        Return
        ------
        fig : object
            Matplotlib figure object for the full figure.
        ax : object
            Matplotlib axes object for the image axes.
        ext : (4) array
            Extent of the image for imshow.
        """
        width = 4.85
        height = 0.8 * self.height / self.width * width
        fig = plt.figure(figsize=(width, height), dpi=300)
        ax = plt.subplot()
        if cal:
            ax.set_xlabel(r"$x$ (mm)")
            ax.set_ylabel(r"$y$ (mm)")
            ext = self.cal * np.array([-0.5, self.width + 0.5, self.height + 0.5, -0.5])
        else:
            ax.set_xlabel(r"$x$ (px)")
            ax.set_ylabel(r"$y$ (px)")
            ext = np.array([-0.5, self.width + 0.5, self.height + 0.5, -0.5])
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        return fig, ax, ext

    def plot_image(self, cal=True, cmap="inferno", metadata=True):
        """Convenient plotting code to quickly look at images with meta data.

        Parameters
        ----------
        cal : bool
            True to show the calibrated axes or false for the pixel coordinates.

        Returns
        -------
        fig : object
            Matplotlib figure object for the full figure.
        ax : object
            Matplotlib axes object for the image axes.
        im : object
            Matplotlib imshow object for the actual image.
        cb : object
            Matplotlib colorbar object.
        ext : (4) array
            Extent of the image for imshow.
        """
        fig, ax, ext = self.create_fig_ax(cal)
        if metadata:
            self.plot_dataset_text(ax)
            self.plot_metadata_text(ax)
        im = ax.imshow(self.data, extent=ext, cmap=cmap)
        cb = fig.colorbar(im, aspect=30 * self.height / self.width)
        cb.set_label("Counts")
        ax.tick_params(color="w")
        ax.spines["bottom"].set_color("w")
        ax.spines["top"].set_color("w")
        ax.spines["left"].set_color("w")
        ax.spines["right"].set_color("w")
        return fig, ax, im, cb, ext

    def plot_dataset_text(self, ax):
        """Add text at the top of the figure stating the dataset and shot number."""
        pass

    def plot_metadata_text(self, ax):
        """Add text at the bottom of the figure stating image parameters."""
        dy = 0.05
        y = 0.02
        # ax.text(0.02, y+dy, r"Size:  {:4d}, {:04d}".format(self.width, self.height), color='w', transform=ax.transAxes)
        # ax.text(0.02, y, r"Start: {:4d}, {:4d}".format(self.offset[0], self.offset[1]), color='w', transform=ax.transAxes)
        # ax.text(0.66, y+dy, r"Exp:  {:0.2f}ms".format(self.shutter), color='w', transform=ax.transAxes)
        # ax.text(0.66, y, r"Gain: {:0.2f}".format(self.gain), color='w', transform=ax.transAxes)
        pass

    def plot_center(self, radius, cal=True, cmap="inferno"):
        """Plot the image with a beam circle and the

        Parameters
        ----------
        radius : float
            Radius of the beam circle to plot.
        cal : bool
            True to show the calibrated axes or false for the pixel coordinates.
        """
        if self.center is None:
            print(
                "The image center is None, it needs to be calculated before it can be shown."
            )
        fig, ax, im, cb, ext = self.plot_image(cal, cmap)
        cen = self.center
        if cal:
            cal = self.cal
        else:
            cal = 1.0
        ax.plot([cal * cen[0], cal * cen[0]], [ext[2], ext[3]], "tab:blue")
        ax.plot([ext[0], ext[1]], [cal * cen[1], cal * cen[1]], "tab:blue")
        phi = np.linspace(0, 2 * np.pi, 1000)
        ax.plot(
            radius * np.cos(phi) + cal * cen[0], radius * np.sin(phi) + cal * cen[1]
        )
        ax.text(0.02, 0.9, "Beam center:", color="tab:blue", transform=ax.transAxes)
        ax.text(
            0.02,
            0.85,
            "({:0.3f}, {:0.3f})".format(cal * cen[0], cal * cen[1]),
            color="tab:blue",
            transform=ax.transAxes,
        )
        return fig, ax, im, cb, ext

    def plot_lineouts(self, cal=True, cmap="inferno"):
        """Plot the image with lineouts through the center.

        Parameters
        ----------
        cal : bool
            True to show the calibrated axes or false for the pixel coordinates.
        """
        if self.center is None:
            print(
                "The image center is None, it needs to be calculated before it can be shown."
            )
        fig, ax, im, cb, ext = self.plot_image(cal, cmap)
        cen = self.center
        if cal:
            cal = self.cal
        else:
            cal = 1.0
        ax.plot([cal * cen[0], cal * cen[0]], [ext[2], ext[3]], "tab:blue")
        ax.plot([ext[0], ext[1]], [cal * cen[1], cal * cen[1]], "tab:blue")
        # TODO implement the lineout bit
        return fig, ax, im, cb, ext


class Elog(IMAGE):
    def __init__(self, camera, date, timestamp):
        self.date = date
        self.timestamp = timestamp
        super().__init__(camera)

    def load_image(self):
        """Load an image from a given data set.

        Returns
        -------
        image : obj
            Pillow image object for the tiff.
        """
        self.path = load.getExternalDataPath()
        self.filename = "ProfMon-CAMR_{!s}-{!s}-{!s}.mat".format(
            self.camera, self.date, self.timestamp
        )
        name = os.path.join(self.path, self.filename)
        try:
            self.mat = scipy.io.loadmat(name)
            image = Image.fromarray(self.mat["data"][0][0][1])
        except (NotImplementedError, ValueError):
            self.mat = None
            self.h5 = h5py.File(name, "r")
            image = Image.fromarray(np.transpose(np.array(self.h5["data"]["img"])))
        return image

    def get_image_meta(self):
        """Return the meta data dictionary from a pillow image object.

        Returns
        -------
        meta : dict
            The meta data dictionary contained in the tiff image.
        """
        # You can see the name of each field in the mat at mat['data'].dtype.names
        meta = {}
        if self.mat is not None:
            mat_meta = self.mat["data"][0][0]
            raise NotImplementedError
            names = self.mat["data"].dtype.names
            ind = names.index("roiX")
            meta["Width"] = meta[ind]
            meta["Offset"] = [mat_meta[10][0][0], mat_meta[11][0][0]]
            meta["Camera"] = self.camera
            meta["Pixel"] = [mat_meta[6][0][0], mat_meta[7][0][0]]
        else:
            h5_meta = self.h5["data"]
            meta["Width"] = int(h5_meta["nCol"][0][0])
            meta["Height"] = int(h5_meta["nRow"][0][0])
            meta["roiX"] = int(h5_meta["roiX"][0][0])
            meta["roiY"] = int(h5_meta["roiY"][0][0])
            meta["Timestamp"] = h5_meta["ts"][0][0]
            meta["pulseId"] = h5_meta["pulseId"][0][0]
            meta["bitdepth"] = int(h5_meta["bitdepth"][0][0])
            meta["orientX"] = h5_meta["orientX"][0][0]
            meta["orientY"] = h5_meta["orientY"][0][0]
        return meta


class DAQ(IMAGE):
    def __init__(self, camera, dataset, filename, ind, step):
        self.dataset = dataset
        self.filename = filename
        self.ind = ind
        self.step = step
        super().__init__(camera)
        if self.meta["Y_ORIENT"] == "Negative" and self.meta["X_ORIENT"] == "Negative":
            self.rotate(180)

    def load_image(self):
        """Load an image from a given data set.

        Returns
        -------
        image : obj
            Pillow image object for the tiff.
        """
        self.path = os.path.join(self.dataset.datasetPath, "images", self.camera)
        name = os.path.join(self.path, self.filename)
        image = Image.open(name)
        return image

    def get_image_meta(self):
        return self.dataset.metadata[self.camera]

    def plot_dataset_text(self, ax):
        """Add text at the top of the figure stating the dataset and shot number."""
        ax.text(
            0.02,
            0.95,
            "{}, Dataset: {}".format(self.dataset.experiment, self.dataset.number),
            color="w",
            transform=ax.transAxes,
        )
        ax.text(
            0.82,
            0.95,
            "Shot: {:04d}".format(self.ind),
            color="w",
            transform=ax.transAxes,
        )
        ax.text(
            0.82,
            0.9,
            "Step: {:02d}".format(self.step),
            color="w",
            transform=ax.transAxes,
        )

    def plot_metadata_text(self, ax):
        """Add text at the bottom of the figure stating image parameters."""
        s = mpl.rcParams["xtick.labelsize"]
        dy = 0.04
        y = 0.02
        sx = self.meta["MinX_RBV"]
        sy = self.meta["MinY_RBV"]
        exp = self.meta["AcquireTime_RBV"]
        gain = self.meta["Gain_RBV"]
        ax.text(
            0.02,
            y + dy,
            "Size:  {:4d}, {:04d}".format(self.width, self.height),
            color="w",
            size=s,
            transform=ax.transAxes,
        )
        ax.text(
            0.02,
            y,
            "Start: {:4d}, {:4d}".format(sx, sy),
            color="w",
            size=s,
            transform=ax.transAxes,
        )
        ax.text(
            0.84,
            y + dy,
            "Exp:  {:0.2f}ms".format(exp * 1e3),
            color="w",
            size=s,
            transform=ax.transAxes,
        )
        ax.text(
            0.84,
            y,
            "Gain: {:0.2f}".format(gain),
            color="w",
            size=s,
            transform=ax.transAxes,
        )
