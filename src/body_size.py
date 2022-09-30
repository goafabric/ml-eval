import pandas as panda
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import matplotlib.pylab as plt
import seaborn as sns


def read_file():
    return panda.read_csv('../csv/body_size.csv')


def train_me(file):
    values = file.values[:, 0:1]
    target = file.values[:, 1:2]

    values_train, values_test, target_train, target_test = train_test_split(values, target, test_size=0.4)

    clf = neighbors.KNeighborsClassifier(1)
    clf.fit(values_train, target_train.ravel())

    print("normal train score", clf.score(values_train, target_train))
    print("normal test score", clf.score(values_test, target_test))
    return clf


df = read_file()
classifier = train_me(df)

print("preedict 180: ", classifier.predict([[180]]))

#print(df.describe())

#df.hist(figsize=(5, 5))
#plt.show()

#df.groupby('type').hist()
#plt.show()

#corrmat = df.corr()
#sns.heatmap(corrmat, annot=True)
#plt.show()
