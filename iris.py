# load iris data and load 0,1,2,4 from matrix/map
from sklearn import datasets
iris = datasets.load_iris()
# this is the input data as a complex matrix with sepal + petal, width/length
X = iris.data
# this is the classification data which can be 0,1,2 base on the 3 possible types
y = iris.target

# X_sepal_length = X[:, 0]
# X_sepal_width = X[:, 1]
# X_petal_length = X[:, 2]
# X_petal_width = X[:, 3]

# split iris data into 60% training and 40% test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# nearest neighbour estimator
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(1)

# train the model
clf.fit(X_train, y_train)

# predict a concrete example
p = clf.predict([[6.3, 2.7, 5.5, 1.5]])
print("preedict", p)
print("")

# normal scores
print("normal train score", clf.score(X_train, y_train))
print("normal test score", clf.score(X_test, y_test))
print("")

# training with sepal only
###
X_train_sepal_only = X_train[:, :2]
X_test_sepal_only = X_test[:, :2]

clf_sepal_10 = neighbors.KNeighborsClassifier(10)
clf_sepal_10.fit(X_train_sepal_only, y_train)

print("sepal only train score: ", clf_sepal_10.score(X_train_sepal_only, y_train))
print("sepal only test score: ",clf_sepal_10.score(X_test_sepal_only, y_test))
print("")

# training with petal only
X_train_petal_only = X_train[:, 2:]
X_test_petal_only = X_test[:, 2:]

clf_petal_10 = neighbors.KNeighborsClassifier(10)
clf_petal_10.fit(X_train_petal_only, y_train)

print("petal only train score: ", clf_petal_10.score(X_train_petal_only, y_train))
print("petal only test score: ", clf_petal_10.score(X_test_petal_only, y_test))
