import pandas as panda
from sklearn.model_selection import train_test_split
from sklearn import neighbors


def read_file():
    return panda.read_csv('doc/body_size_short.csv')


def train_me():
    X = df.values[:, 0:1]
    y = df.values[:, 1:2]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    clf = neighbors.KNeighborsClassifier(1)
    clf.fit(x_train, y_train.ravel())

    print("normal train score", clf.score(x_train, y_train))
    print("normal test score", clf.score(x_test, y_test))
    return clf


df = read_file()
clfn = train_me()
p = clfn.predict([[54]])

print("preedict", p)

