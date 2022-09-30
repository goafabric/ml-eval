import pandas as panda
import matplotlib.pylab as plt

def read_file():
    return panda.read_csv('../../csv/iris_dirty.csv',
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


def convert_from_mm(row):
    return panda.to_numeric(row['petal width'].replace(' mm', '')) / 10


def normalize_measures():
    print(df.head())
    df['petal width'] = df.apply(convert_from_mm, axis='columns')
    print(df.head())


df = read_file()

search_missing_sepal()
calculate_missing_sepal()

eliminate_duplicate()
eliminate_typo()

normalize_measures()

#hist
print(df.describe())


df.hist(figsize=(15, 15))
# plt.show()

# df.groupby('class').describe()
df.groupby('class').hist()
# plt.show()

#seaborn
import seaborn as sns
# sns.jointplot(df['sepal length'], df['petal length'])
# plt.show()


corrmat = df.corr()
sns.heatmap(corrmat, annot=True)
# plt.show()



