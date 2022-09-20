# load iris data and load 0,1,2,4 from matrix/map
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors

iris = datasets.load_iris()
# this is the input data as a complex matrix with sepal + petal, width/length
X = iris.data
# this is the classification data which can be 0,1,2 base on the 3 possible types
y = iris.target

# split iris data into 60% training and 40% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


def train_normal():
    # nearest neighbour estimator
    clf = neighbors.KNeighborsClassifier(1)
    # train the model
    clf.fit(X_train, y_train)
    return clf


clf = train_normal()


def predict_concrete_example():
    p = clf.predict([[6.3, 2.7, 5.5, 1.5]])
    print("preedict", p)
    print("")


def print_normal_scores():
    print("normal train score", clf.score(X_train, y_train))
    print("normal test score", clf.score(X_test, y_test))
    print("")


def training_sepal_only():
    X_train_sepal_only = X_train[:, :2]
    X_test_sepal_only = X_test[:, :2]
    clf_sepal_10 = neighbors.KNeighborsClassifier(10)
    clf_sepal_10.fit(X_train_sepal_only, y_train)
    print("sepal only train score: ", clf_sepal_10.score(X_train_sepal_only, y_train))
    print("sepal only test score: ", clf_sepal_10.score(X_test_sepal_only, y_test))
    print("")


def training_petal_only():
    X_train_petal_only = X_train[:, 2:]
    X_test_petal_only = X_test[:, 2:]
    clf_petal_10 = neighbors.KNeighborsClassifier(10)
    clf_petal_10.fit(X_train_petal_only, y_train)
    print("petal only train score: ", clf_petal_10.score(X_train_petal_only, y_train))
    print("petal only test score: ", clf_petal_10.score(X_test_petal_only, y_test))


predict_concrete_example()
print_normal_scores()

training_sepal_only()
training_petal_only()
