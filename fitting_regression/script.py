# Copyright for the LoadCsv function
__copyright__ = 'Copyright (C) 2023-2024 Taylor Bybee'

import numpy as np

def LoadCsv(fn: str):
    """
    This function loads a CSV file containing latitude and longitude.
    """
    arr = np.loadtxt(fn, delimiter=',', dtype=np.float64)
    return arr


def describe_data():
    # import all of the csv files and print size of each one
    base = 'data/data_03_'
    for i in ['a', 'b', 'c', 'd', 'e', 'f']:
        file = base + i + '.csv'
        xy = LoadCsv(file)
        first_row = xy[0]
        
        print("File name: " + file)
        print("Data points: " + str(xy.shape[0]))
        print("Data first row: [" + ', '.join(map(str, first_row)) + "] \n")    
    return

def main():
    describe_data()
    return

# Entry Point
if __name__ == '__main__':
    main()