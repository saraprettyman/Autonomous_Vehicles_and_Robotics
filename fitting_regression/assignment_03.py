__author__ = 'Taylor Bybee'
__copyright__ = 'Copyright (C) 2023-2024 Taylor Bybee'

# Example terminal command: python assignment_03.py data/data_03_a.csv

# Imports
import numpy as np
import argparse
import math
import matplotlib.pyplot as plt
from sklearn import linear_model
import time

def LoadCsv(fn: str):
    """
    This function loads a CSV file containing latitude and longitude.
    """
    arr = np.loadtxt(fn, delimiter=',', dtype=np.float64)
    return arr


def RansacLine(xy, i=None):
    reg = linear_model.RANSACRegressor(random_state=0, residual_threshold=i).fit(xy[:,0].reshape(-1, 1), xy[:,1])
    y_pred = reg.predict(xy[:,0].reshape(-1, 1))
    return y_pred

def HuberLine(xy):
    reg = linear_model.HuberRegressor(epsilon=1.0, max_iter=10000).fit(xy[:,0].reshape(-1, 1), xy[:,1])
    y_pred = reg.predict(xy[:,0].reshape(-1, 1))
    return y_pred

def LeastSquaresLine(xy):
    reg = linear_model.LinearRegression().fit(xy[:,0].reshape(-1, 1), xy[:,1])
    y_pred = reg.predict(xy[:,0].reshape(-1, 1))
    return y_pred

def ParseArgs():
    """
    This function parses arguments to the program.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    parser.add_argument('-i', default=None, type=float)
    args = parser.parse_args()
    return args


def main():
    # Parse Arguments
    args = ParseArgs()

    # Load data
    xy = LoadCsv(args.csv)
    print(f'Loaded {xy.shape[0]} data points.')

    # Fit lines
    t_start = time.time()
    ransac_y = RansacLine(xy, args.i)
    t_end = time.time()
    print(f'RANSAC took {(t_end - t_start)*1000} milliseconds with RANSAC threshold {args.i}.')

    t_start = time.time()
    huber_y = HuberLine(xy)
    t_end = time.time()
    print(f'Huber took {(t_end - t_start)*1000} milliseconds.')

    t_start = time.time()
    ls_y = LeastSquaresLine(xy)
    t_end = time.time()
    print(f'Least-Squares took {(t_end - t_start)*1000} milliseconds.')

    # Now we want to plot the data!
    fig, ax = plt.subplots(2, 2)

    for i in range(2):
        for j in range (2):
            ax[i, j].scatter(xy[:,0], xy[:,1], s=0.5, label='Original Points')
            ax[i, j].legend()
            ax[i, j].set_aspect('equal', adjustable='datalim')
            ax[i, j].grid()

    # Plot all three together
    ax[0, 0].plot(xy[:,0], ls_y, 'b', label='LS Line')
    ax[0, 0].plot(xy[:,0], huber_y, 'm', label='Huber Line')
    ax[0, 0].plot(xy[:,0], ransac_y, 'r', label='RANSAC Line')
    ax[0, 0].legend()


    ax[0, 1].plot(xy[:,0], ls_y, 'b', label='Line')
    ax[0, 1].set_title('Least Squares Line')

    ax[1, 0].plot(xy[:,0], huber_y, 'm', label='Line')
    ax[1, 0].set_title('Huber Line')

    ax[1, 1].plot(xy[:,0], ransac_y, 'r', label='Line')
    ax[1, 1].set_title('RANSAC Line')

    plt.show()


# Entry Point
if __name__ == '__main__':
    main()