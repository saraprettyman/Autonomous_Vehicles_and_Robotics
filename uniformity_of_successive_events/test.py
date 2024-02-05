import pandas as pd

# Load csv data
data_a = pd.read_csv('data/data_02_a.csv')
data_b = pd.read_csv('data/data_02_b.csv')
data_c = pd.read_csv('data/data_02_c.csv')
data_d = pd.read_csv('data/data_02_d.csv')

# Describe each csv file
def describe_data(data, name):
    print("File name: ", name)
    print("Data shape: ", data.shape)
    print("Data first value:", data.iloc[0].values, "\n")

# Describe each csv file
describe_data(data_a, 'data_02_a.csv')
describe_data(data_b, 'data_02_b.csv')
describe_data(data_c, 'data_02_c.csv')
describe_data(data_d, 'data_02_d.csv')