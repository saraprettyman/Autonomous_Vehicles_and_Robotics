''' Prototyped Metrics and Algorithms
 data: csv file converted to an array
 output: boolean value of 0 and 1, bad and good respectively
'''
import pandas as pd


def kss(data):
    '''
    input: 
    '''
    return 0

def max_diff(data):
    return 0

def min_diff(data):
    return 0

def med_diff(data):
    return 0

def mean_diff(data):
    return 0

def difference(data):
    i = 1
    l = len(data)
    diff = []

    while i < l:
        d = (data.iloc[i].values-data.iloc[i-1].values)
        diff.append(d)
        i += 1
    return diff

data = pd.read_csv('data/data_02_b.csv')
print(difference(data))
