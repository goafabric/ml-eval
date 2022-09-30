import pandas as panda
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import matplotlib.pylab as plt
import seaborn as sns


def read_file():
    return panda.read_csv('../csv/tv_price_fake.csv')


def train_me(file):
    values = file.values[:, 0:2]
    target = file.values[:, 2:3]

    values_train, values_test, target_train, target_test = train_test_split(values, target, test_size=0.4)

    clf = neighbors.KNeighborsClassifier(1)
    clf.fit(values_train, target_train.ravel())

    print("normal train score", clf.score(values_train, target_train))
    print("normal test score", clf.score(values_test, target_test))
    return clf


df = read_file()
classifier = train_me(df)
prediction = classifier.predict([[14, 3599]])

print("preedict 4599: ", classifier.predict([[14, 4599]]))
print("preedict 2599: ", classifier.predict([[14, 2599]]))
print("preedict 620: ", classifier.predict([[6, 620]]))


#print(df.describe())

#df.hist(figsize=(10, 10))
#plt.show()

#corrmat = df.corr()
#sns.heatmap(corrmat, annot=True)
#plt.show()
