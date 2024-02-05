__author__ = 'Taylor Bybee'
__copyright__ = 'Copyright (C) 2023 Taylor Bybee'

# Imports
import numpy as np
import argparse
import matplotlib.pyplot as plt

def LoadTimestamps(fn: str):
    """
    This function loads a CSV file containing timestamps.
    """
    arr = np.loadtxt(fn, delimiter=',', dtype=np.float64)
    arr = arr - arr[0]
    return arr[:]

def ParseArgs():
    """
    This function parses arguments to the program.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    parser.add_argument('-w', '--window-size', default=1.0, type=float)
    parser.add_argument('-a', '--algorithm-rate-hz', default=10.0, type=float)

    args = parser.parse_args()
    return args

def ComputeDiffStats(stamps):
    """
    This function takes an np array of timestamps and computes
    statistics regarding the differences between successive
    timestamps.
    """
    delta_min = float('nan')
    delta_max = float('nan')
    delta_median = float('nan')
    delta_mean = float('nan')

    if len(stamps) > 1:
        diffs = np.diff(stamps)
        delta_max = np.max(diffs)
        delta_min = np.min(diffs)
        delta_median = np.median(diffs)
        delta_mean = np.mean(diffs)

    return (delta_min, delta_max, delta_median, delta_mean)


def ComputeKolmogorovSmirnovStatistic(stamps, start, end):
    """
    This function takes an np array of timestamps and computes
    the Kolmogorov-Smirnov statistic based on the fact that 
    the samples should follow a uniform distribution over the
    time window specified.
    """
    D = 1.0 # Default to worst-case if there is no data

    if len(stamps) > 0:
        D = 0.0
        window_size_seconds = end - start
        current_sample_cdf = 0.0
        sample_contribution = 1.0 / len(stamps)
        for elem in stamps:
            seconds_from_start = elem - start

            # Theoretical Uniform CDF Value
            theoretical_cdf_value = seconds_from_start / window_size_seconds

            # Compute the actual CDF Value
            # This is done twice to get both "sides" of the stair-step
            D = max(D, abs(theoretical_cdf_value - current_sample_cdf))
            current_sample_cdf += sample_contribution # increment the CDF
            D = max(D, abs(theoretical_cdf_value - current_sample_cdf))

        # Last comparison to get the final CDF value
        D = max(D, abs(1.0 - current_sample_cdf))

        # TO DO - Can we animate the theoretical CDF vs the realized CDF?

    return D

def main():
    # Parse Arguments
    args = ParseArgs()

    # Load data and set starting time to 0.0
    timestamps = LoadTimestamps(args.csv)

    # Run the Kolmogorov Algorithm
    start_time = timestamps[0]
    end_time = timestamps[-1]
    print(f'Start & End Times: [{start_time}, {end_time}]. Window Size: {args.window_size}s.')


    # Simulate time steps
    dt_step = 1.0 / args.algorithm_rate_hz
    t = start_time
    results = np.empty(shape=(0,6))
    while t <= end_time:

        # Get the elements within the window
        query_start = t - args.window_size
        query_end = t
        samples_in_window = timestamps[np.where(np.logical_and(timestamps>=query_start, timestamps<=query_end))]

        # Compute the KolmogorovSmirnov Statistic
        ks_stat = ComputeKolmogorovSmirnovStatistic(samples_in_window, query_start, query_end)
        delta_min, delta_max, delta_median, delta_mean = ComputeDiffStats(samples_in_window)

        # Add results to results array
        newrow = [t, ks_stat, delta_min, delta_max, delta_median, delta_mean]
        results = np.vstack([results, newrow])

        # Increment the time step for next time
        t = t + dt_step

    # Now we want to plot the data!
    fig, ax = plt.subplots(2, 1, sharex=True)

    # KS Score
    ax[0].plot(results[:, 0], results[:,1], 'm', label='Kolmogorov-Smirnov')
    ax[0].plot(timestamps, np.zeros(timestamps.shape), 'k*', label='Event Samples')
    ax[0].grid()
    ax[0].set_ylabel('Kolmogorov-Smirnov Score')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_title(f'Kolmogorov-Smirnov Score over Time Window {args.window_size}s')
    leg0 = ax[0].legend(loc ="best")

    # Deltas
    ax[1].plot(results[:, 0], results[:, 2], 'm', label='Delta Minimum')
    ax[1].plot(results[:, 0], results[:, 3], 'r', label='Delta Maximum')
    ax[1].plot(results[:, 0], results[:, 4], 'c', label='Delta Median')
    ax[1].plot(results[:, 0], results[:, 5], 'b', label='Delta Mean')
    ax[1].plot(timestamps, np.zeros(timestamps.shape), 'k*', label='Event Samples')
    ax[1].grid()
    ax[1].set_ylabel('Diff Stats (s)')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_title(f'Diffs Stats over Time Window {args.window_size}s')
    leg1 = ax[1].legend(loc ="best")

    plt.show()

# Entry Point
if __name__ == '__main__':
    main()