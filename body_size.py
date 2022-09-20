import pandas as panda
from sklearn.model_selection import train_test_split
from sklearn import neighbors


def read_file():
    return panda.read_csv('doc/body_size.csv')


df = read_file()

X = df.values[:, 0:1]
y = df.values[:, 1:2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

clf = neighbors.KNeighborsClassifier(1)
clf.fit(X_train, y_train.ravel())

p = clf.predict([[180]])

print("normal train score", clf.score(X_train, y_train))
print("normal test score", clf.score(X_test, y_test))
print("")
print("preedict", p)

