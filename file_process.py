import pandas as pd


def read_file():
    return pd.read_csv('doc/iris_dirty.csv',
                       header=None,
                       names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])


def calculate_missing_sepal():
    iris_versicolor = df[df['class'] == 'Iris-versicolor']
    mean_sepal_width = pd.Series.mean(iris_versicolor['sepal width'])
    df.loc[82, 'sepal width'] = mean_sepal_width


df = read_file()

print('looking for missing sepal')
print(df.count())
print(df[df['sepal width'].isnull()])

print('correcting missing')
calculate_missing_sepal()
print(df.count())
