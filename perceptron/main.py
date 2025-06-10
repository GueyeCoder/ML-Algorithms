import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from perceptron import Perceptron

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_pred)
    return accuracy

X, y = datasets.make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=1.05, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

p = Perceptron()
p.fit(X_train, y_train)
predictions = p.predict(X_test)

print("Perceptron classification accuracy ", accuracy(y_test, predictions))