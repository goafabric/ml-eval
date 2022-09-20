import pandas as panda


def read_file():
    return panda.read_csv('doc/iris_dirty.csv',
                          header=None,
                          names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])


def search_missing_sepal():
    print('looking for missing sepal')
    print(df.count())
    print(df[df['sepal width'].isnull()])


def calculate_missing_sepal():
    print('correcting missing')
    iris_versicolor = df[df['class'] == 'Iris-versicolor']
    mean_sepal_width = panda.Series.mean(iris_versicolor['sepal width'])
    df.loc[82, 'sepal width'] = mean_sepal_width
    print(df.count())


def eliminate_duplicate():
    print(df[df.duplicated(keep=False)])
    print(df.groupby('class').count())
    df.drop(df.index[[100]])


def eliminate_typo():
    typo_idx = df[df['class'] == 'Iris-setsoa'].index
    df.loc[typo_idx, 'class'] = 'Iris-setosa'
    print(df.groupby('class').count())


df = read_file()

search_missing_sepal()
calculate_missing_sepal()

eliminate_duplicate()
eliminate_typo()

# measures
print(df.head())
