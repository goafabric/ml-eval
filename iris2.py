from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
Y = iris.target

Y_sepal_length = X[:, 0]
Y_sepal_width = X[:, 1]

print(Y_sepal_length)



