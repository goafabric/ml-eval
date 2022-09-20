import pandas as panda
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors


def read_file():
    return panda.read_csv('doc/iris_cleansed.csv')


def convert_from_mm(row):
    return panda.to_numeric(row['class']
                            .replace('Iris-setosa', '0')
                            .replace('Iris-versicolor', '1')
                            .replace('Iris-virginica', '2')
                            )


df = read_file()

# df.apply(convert_from_mm, axis='columns')

X = df.values[:, 0:4]
y = df.values[:, 4:5]

#train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

clf = neighbors.KNeighborsClassifier(1)
clf.fit(X_train, y_train.ravel())

p = clf.predict([[6.3, 2.7, 5.5, 1.5]])
print("preedict", p)
