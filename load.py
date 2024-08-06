import os
import requests
import subprocess


__ElogBaseURL = "http://mccas0.slac.stanford.edu/u1/facet/matlab/data/"
__SLACLinuxPath = "/nfs/slac/g/facetdata/nas/nas-li20-pm00"
__ExternalDataPath = "B-ExternalData"
dataPath = None
SLACUser = None


def ElogImage(cam: str, date: str, time: str):
    """Loads the image matching the given name from the Elog.

    Automatically copies the image into the B-ExternalData directory.
    If the image already exists in B-ExternalData directory, this function does nothing.

    Args:
        cam: Camera designator to load data, e.g. 'LI20_111'.
        date: Date string for when the image was taken in YYYY-MM-DD: '2022-10-27'.
        time: Time stamp for the requested image in HHMMSS: '175546'.
    """
    # Check if the image already exists in B-ExternalData
    filename = "ProfMon-CAMR_{}-{}-{}.mat".format(cam, date, time)
    filepath = os.path.join(__ExternalDataPath, filename)
    if not os.path.exists(__ExternalDataPath):
        os.makedirs(__ExternalDataPath)
    if os.path.exists(filepath):
        return

    # Try and connect to the SLAC servers, throw an error if they aren't available
    year, month, day = parseDate(date)
    url = __ElogBaseURL + "{}/{}-{}/{}-{}-{}/{}".format(
        year, year, month, year, month, day, filename
    )
    try:
        response = requests.get(url)
        open(filepath, "wb").write(response.content)
    except:
        raise RuntimeError(
            "The file was not already downloaded and something went wrong trying to download it, are you on the VPN?"
        )


def dataset(experiment, date, number):
    """Loads the dataset for the given experiment and number from SLAC.

    Copies all of the data to the local data directory.
    If the dataset already exists in the local data directory, this function does nothing.

    Args:
        experiment: SLAC experiment number, e.g. 'E301'.
        date: Date string for when the dataset was taken in YYYY-MM-DD: '2022-10-27'.
        number: ID number of the dataset, printed in the Elog entry.
    """
    # Check if the dataset already exists in the local data directory
    if dataPath is None:
        raise RuntimeError(
            "dataPath was not set in the load module, it does not know where to download data!"
        )
    experimentPath = os.path.join(dataPath, experiment)
    datasetPath = os.path.join(dataPath, experiment, "{}_{}".format(experiment, number))
    if os.path.exists(datasetPath):
        return
    else:
        if not os.path.exists(experimentPath):
            os.makedirs(experimentPath)

    # Try and connect to the SLAC servers, throw an error if the command fails
    # Don't use OS.path.join - this always accesses a linux file system but it can be run from either a linux or windows file system
    year, month, day = parseDate(date)
    SLACPath = __SLACLinuxPath + "/{}/{}/{}{}{}/{}_{}".format(
        experiment, year, year, month, day, experiment, number
    )
    if os.name == "nt":
        targetPath = winPathToWSLPath(experimentPath)
    else:
        targetPath = datasetPath
    rsyncCommand = "rsync -av {}@centos7.slac.stanford.edu:{} {}".format(
        SLACUser, SLACPath, targetPath
    )
    print("Downloading with command:")
    print(rsyncCommand)
    if os.name == "nt":
        cp = subprocess.run(["powershell", "-Command", "wsl " + rsyncCommand])
        if cp.returncode != 0:
            raise RuntimeError(
                "Command {} failed with exit code {} in the windows powershell.".format(
                    cp.args[2], cp.returncode
                )
            )
    else:
        # TODO implement this once I have a linux computer to develop on
        # Launch a shell and all that good stuff
        pass


def parseDate(date: str) -> tuple[str, str, str]:
    """Returns the year, month, and day from the passed date string.

    Args:
        date: Date string, format: '2022-10-27'.

    Returns:
        year: The year from the date string, format YYYY.
        month: The month from the date string, format MM.
        day: The day from the date string, format DD.
    """
    year, month, day = date.split("-")
    return year, month, day


def winPathToWSLPath(winPath: str) -> str:
    """Returns the equivalent path in WSL for the given windows path.

    Args:
        winPath: Path to a file or directory in windows, must be in windows format ('C:/Users/user/Documents/').

    Returns:
        wslPath: The path to the same file or directory in WSL.
    """
    drive, path = winPath.split(":\\")
    drive = drive.lower()
    path = path.replace("\\", "/")
    return "/mnt/{}/{}".format(drive, path)


def getExternalDataPath() -> str:
    """Returns the external data path.

    Returns:
        path: The path to the folder where Elog images are downloaded.
    """
    return __ExternalDataPath


def srv20_dataset(number):
    """Load the dataset from the number on the controls network."""
    raise NotImplementedError
