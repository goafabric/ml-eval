import matplotlib.pylab as plt
import numpy as np

# load iris data and load 0,1,2,4 from matrix/map
from sklearn import datasets
iris = datasets.load_iris()
#this is the input data as a complex matrix with sepal + petal, width/length
X = iris.data
#this is the classification data which can be 0,1,2 base on the 3 possible types
y = iris.target

#X_sepal_length = X[:, 0]
#X_sepal_width = X[:, 1]
#X_petal_length = X[:, 2]
#X_petal_width = X[:, 3]

# split iris data into 60% training and 40% test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# nearest neighbour estimator
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(1)

#train the model
clf.fit(X_train, y_train)

#predict a concrete example
p = clf.predict([[6.3, 2.7, 5.5, 1.5]])
print("preedict", p)

print("train score", clf.score(X_train, y_train))

print("test score", clf.score(X_test, y_test))

###

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
font_size=25

def meshGrid(x_data, y_data):
    h = .02  # step size in the mesh
    x_min, x_max = x_data.min() - 1, x_data.max() + 1
    y_min, y_max = y_data.min() - 1, y_data.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return (xx,yy)

def plotPrediction(clf, x_data, y_data, x_label, y_label, colors, title="", mesh=True):
    xx,yy = meshGrid(x_data, y_data)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(20,10))
    if mesh:
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.scatter(x_data, y_data, c=colors, cmap=cmap_bold, s=80, marker='o')
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.title(title, fontsize=font_size)

###

X_train_sepal_only = X_train[:, :2]
X_test_sepal_only = X_test[:, :2]

clf_sepal = neighbors.KNeighborsClassifier(1)
clf_sepal.fit(X_train_sepal_only, y_train)

plotPrediction(clf_sepal, X_train_sepal_only[:, 0], X_train_sepal_only[:, 1],
               'Sepal length', 'Sepal width', y_train, mesh=False,
               title="Train Data for Sepal Features")


clf_sepal_10 = neighbors.KNeighborsClassifier(10)
clf_sepal_10.fit(X_train_sepal_only, y_train)

print("sepal only train score: ", clf_sepal_10.score(X_train_sepal_only, y_train))
print("sepal only test score: ",clf_sepal_10.score(X_test_sepal_only, y_test))

plotPrediction(clf_sepal_10, X_train_sepal_only[:, 0], X_train_sepal_only[:, 1],
               'Sepal length', 'Sepal width', y_train,
               title="Model too simple even for Train Data")

###

X_train_petal_only = X_train[:, 2:]
X_test_petal_only = X_test[:, 2:]

clf_petal_10 = neighbors.KNeighborsClassifier(10)
clf_petal_10.fit(X_train_petal_only, y_train)

print("petal only train score: ", clf_petal_10.score(X_train_petal_only, y_train))
print("petal only test score: ", clf_petal_10.score(X_test_petal_only, y_test))

plotPrediction(clf_petal_10, X_train_petal_only[:, 0], X_train_petal_only[:, 1],
               'Petal length', 'Petal width', y_train,
               title="Simple model looks good for Train Data")

#plt.show()