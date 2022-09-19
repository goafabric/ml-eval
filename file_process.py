import pandas as pd


def read_file():
    return pd.read_csv('doc/iris_dirty.csv',
                       header=None,
                       names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])


df = read_file()

# print(df)

print(df.head())

print(df.count())
