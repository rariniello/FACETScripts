import numpy as np

from scipy.constants import physical_constants

c = physical_constants["speed of light in vacuum"][0]
e = physical_constants["elementary charge"][0]
me = physical_constants["electron mass"][0]
eps_0 = physical_constants["vacuum electric permittivity"][0]


def facetSpec():
    spec = {}
    spec["Q0D"] = {
        "s": 1996.98249,
        "l": 1.0,
    }
    spec["Q1D"] = {
        "s": 1999.206615,
        "l": 1.0,
    }
    spec["Q2D"] = {
        "s": 2001.431049,
        "l": 1.0,
    }
    spec["B5D36"] = {
        "s": 2005.940051,
        "l": 1.0,
    }
    spec["CHER"] = 2016.22
    spec["EDC_SCREEN"] = 2010.60
    spec["PEXT"] = 1995.04
    return spec


def Mmat(l, K=None):
    if K is None:
        M = np.array(
            [
                [1.0, l, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, l],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    elif K > 0.0:
        M = np.array(
            [
                [np.cos(np.sqrt(K) * l), np.sin(np.sqrt(K) * l) / np.sqrt(K), 0.0, 0.0],
                [
                    -np.sin(np.sqrt(K) * l) * np.sqrt(K),
                    np.cos(np.sqrt(K) * l),
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    np.cosh(np.sqrt(K) * l),
                    np.sinh(np.sqrt(K) * l) / np.sqrt(K),
                ],
                [
                    0.0,
                    0.0,
                    np.sinh(np.sqrt(K) * l) * np.sqrt(K),
                    np.cosh(np.sqrt(K) * l),
                ],
            ]
        )
    elif K < 0.0:
        K = abs(K)
        M = np.array(
            [
                [
                    np.cosh(np.sqrt(K) * l),
                    np.sinh(np.sqrt(K) * l) / np.sqrt(K),
                    0.0,
                    0.0,
                ],
                [
                    np.sinh(np.sqrt(K) * l) * np.sqrt(K),
                    np.cosh(np.sqrt(K) * l),
                    0.0,
                    0.0,
                ],
                [0.0, 0.0, np.cos(np.sqrt(K) * l), np.sin(np.sqrt(K) * l) / np.sqrt(K)],
                [
                    0.0,
                    0.0,
                    -np.sin(np.sqrt(K) * l) * np.sqrt(K),
                    np.cos(np.sqrt(K) * l),
                ],
            ]
        )
    return M


def set_spec_B(spec, setQ0D, setQ1D, setQ2D, setB5D36):
    """Set magnet k from integrated field strength in kGauss."""
    spec["Q0D"]["B"] = setQ0D / spec["Q0D"]["l"] * 0.1
    spec["Q1D"]["B"] = setQ1D / spec["Q1D"]["l"] * 0.1
    spec["Q2D"]["B"] = setQ2D / spec["Q2D"]["l"] * 0.1
    spec["B5D36"]["E"] = setB5D36


def get_spec_matrix(spec, start, end, gb):
    K_Q0D = spec["Q0D"]["B"] * e / (gb * me * c)
    K_Q1D = spec["Q1D"]["B"] * e / (gb * me * c)
    K_Q2D = spec["Q2D"]["B"] * e / (gb * me * c)

    if isinstance(start, str):
        s_start = spec[start]
    else:
        s_start = start
    s_end = spec[end]
    l_D0 = (spec["Q0D"]["s"] - 0.5 * spec["Q0D"]["l"]) - s_start
    l_D1 = (spec["Q1D"]["s"] - 0.5 * spec["Q1D"]["l"]) - (
        spec["Q0D"]["s"] + 0.5 * spec["Q0D"]["l"]
    )
    l_D2 = (spec["Q2D"]["s"] - 0.5 * spec["Q2D"]["l"]) - (
        spec["Q1D"]["s"] + 0.5 * spec["Q1D"]["l"]
    )
    l_D3 = s_end - (spec["Q2D"]["s"] + 0.5 * spec["Q2D"]["l"])

    M_D0 = Mmat(l_D0, K=None)
    M_Q0D = Mmat(spec["Q0D"]["l"], K_Q0D)
    M_D1 = Mmat(l_D1, K=None)
    M_Q1D = Mmat(spec["Q1D"]["l"], K_Q1D)
    M_D2 = Mmat(l_D2, K=None)
    M_Q2D = Mmat(spec["Q2D"]["l"], K_Q2D)
    M_D3 = Mmat(l_D3, K=None)
    M = np.matmul(
        M_D3,
        np.matmul(
            M_Q2D,
            np.matmul(M_D2, np.matmul(M_Q1D, np.matmul(M_D1, np.matmul(M_Q0D, M_D0)))),
        ),
    )
    return M


def get_dispersion(spec, location):
    D_CHER = 60.0
    dz_CHER = spec["CHER"] - spec["B5D36"]["s"]
    dz = spec[location] - spec["B5D36"]["s"]
    if dz < 0.0:
        return 0.0
    D = (spec["B5D36"]["E"] / 10.0) * (D_CHER / dz_CHER) * dz
    return D
