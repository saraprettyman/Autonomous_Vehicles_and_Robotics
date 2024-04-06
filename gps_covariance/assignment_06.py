__author__ = 'Taylor Bybee'
__copyright__ = 'Copyright (C) 2023-2024 Taylor Bybee'

# Imports
import numpy as np
import argparse
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# This function was taken from: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def LoadCsv(fn: str):
    """
    This function loads a CSV file containing latitude and longitude.
    """
    arr = np.loadtxt(fn, delimiter=',', dtype=np.float64)
    return arr


def ConvertLatLonToXY(lat_lon_alt: np.array, origin: np.array):
    """
    This function converts lat lon alt to XY using Gnomonic Projection.
    """
    # Gnomonic Projection Constants
    EARTH_RADIUS = 6378137.0 # In meters
    PI = 3.14159265358979323846  # We define our own pi here to control the value.
    DEG_TO_RAD = PI / 180.0

    baseLat = origin[0] * DEG_TO_RAD
    baseLon = origin[1] * DEG_TO_RAD
    sinBaseLat = math.sin(baseLat)
    cosBaseLat = math.cos(baseLat)
    distanceToEarthCenter = origin[2] + EARTH_RADIUS

    n_samples = lat_lon_alt.shape[0]
    xy = np.zeros((n_samples, 2))

    for i in range(n_samples):
        latitude = lat_lon_alt[i, 0]
        longitude = lat_lon_alt[i, 1]

        inLat = latitude * DEG_TO_RAD
        inLon = longitude * DEG_TO_RAD
        lonDiff = inLon - baseLon
        sinLat =     math.sin(inLat)
        cosLat =     math.cos(inLat)
        sinLon =     math.sin(lonDiff)
        cosLon =     math.cos(lonDiff)

        x = distanceToEarthCenter / (sinBaseLat * sinLat + cosBaseLat * cosLat * cosLon)
        y = x

        x *= cosLat * sinLon
        y *= cosBaseLat * sinLat - sinBaseLat * cosLat * cosLon

        xy[i, 0] = x
        xy[i, 1] = y

    return xy


def ParseArgs():
    """
    This function parses arguments to the program.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    args = parser.parse_args()
    return args


def main():
    # Parse Arguments
    args = ParseArgs()

    # Load data
    lat_lon_alt = LoadCsv(args.csv)
    print(f'Loaded {lat_lon_alt.shape[0]} data points.')

    # Find the mean of latitude and longitude
    origin = np.mean(lat_lon_alt, axis=0)
    print(f'Origin computed at ({origin[0]}, {origin[1]}, {origin[2]})')

    # Convert Lat Lon to XY
    xy_unaligned = ConvertLatLonToXY(lat_lon_alt, origin)

    # Rotate the XY to align with the axes. The XY should be 0-centered already, so no need to translate.
    U, S, Vh = np.linalg.svd(np.transpose(xy_unaligned), full_matrices=True)
    xy = np.matmul(xy_unaligned, np.transpose(U))

    # Compute the covariance
    c = np.ma.cov(np.transpose(xy))
    print('Covariance Matrix:\n', c)
    print(f'Primary Standard Deviation: {math.sqrt(c[0,0])} meters.')
    print(f'Secondary Standard Deviation: {math.sqrt(c[1,1])} meters.')

    # Now we want to plot the data!
    fig, ax = plt.subplots(figsize=(6, 6))

    mu = 0, 0
    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)

    x = xy[:,0]
    y = xy[:,1]

    ax.scatter(x, y, s=0.5)
    confidence_ellipse(x, y, ax, n_std=1,
                    label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x, y, ax, n_std=2,
                    label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(x, y, ax, n_std=3,
                    label=r'$3\sigma$', edgecolor='blue', linestyle=':')

    ax.scatter(mu[0], mu[1], c='red', s=3)
    ax.set_title('Projected Data & Covariance Ellipses')
    ax.legend()
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid()
    plt.show()


# Entry Point
if __name__ == '__main__':
    main()