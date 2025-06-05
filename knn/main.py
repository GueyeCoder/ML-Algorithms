
from collections import Counter

import numpy as np 
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                          n_redundant=10, n_classes=3, random_state=42)
# iris = datasets.load_iris()
# X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


# print(X_train.shape)
# print(X_train[0])
# print(y_train.shape)
# print(y_test)

# plt.figure()
# plt.scatter(X[:, 0], X[:,2], c=y, cmap=cmap, edgecolors='k', s=20)
# plt.show()

from knn import KNN
clf = KNN(k=3)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)