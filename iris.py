import warnings
warnings.filterwarnings('ignore')

import matplotlib.pylab as plt
import numpy as np

import sklearn
print(sklearn.__version__)

from sklearn.datasets import load_iris
iris = load_iris()

print(iris.DESCR)

X = iris.data
y = iris.target

print(X)
print(y)


X.shape, y.shape
X[0]

print(X[0])