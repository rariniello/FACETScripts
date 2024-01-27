import numpy as np


def EFieldFromImage(inputIM, size, cal):
    """Takes the image from a camera viewing the laser and converts it to an electric field.

    Use this function to convert an image of the laser into an numpy array suitable
    for feeding into fourierProp. Often it is useful to first create a centered image
    using image.get_center_image. Selects a square around the center of width 2*size.

    Args:
        inputIM: Numpy array representing the input image.
        size: Halfwidth of the square output [px].
        cal: Calibration for the camera [mm/px].

    Returns:
        E: A numpy array representing the electric field. The array is the sqrt of the
            camera counts.
        XD: Domain size of the grid, to feed into the simulation grid.
        NxD: The size of the transverse grid, to feed into the simulation grid.
    """

    size = 300
    shape = np.shape(inputIM)
    image_in = inputIM[
        int(0.5 * shape[0] - size) : int(0.5 * shape[0] + size),
        int(0.5 * shape[1] - size) : int(0.5 * shape[1] + size),
    ]

    xp = np.arange(0, shape[0], 1)
    x = cal * xp
    x_in = x[int(0.5 * shape[0] - size) : int(0.5 * shape[0] + size)]
    dx = x_in[1] - x_in[0]
    XD = (x_in[-1] - x_in[0] + dx) * 1e-3
    NxD = len(x_in)

    E = np.rot90(np.sqrt(abs(image_in)), 3)
    return E, XD, NxD


def extentFromGrid(x, y):
    """Returns the extent needed for matplotlib imshow from the x and y grids.

    The data is assumed to have x along the first axis (rows) and y along the second axis (cols).

    Args:
        x: Corrdinates of each point in the x direction.
        y: Coordinates of each point in the y direction
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return np.array(
        [x[0] - 0.5 * dx, x[-1] + 0.5 * dx, y[0] - 0.5 * dy, y[-1] + 0.5 * dy]
    )
