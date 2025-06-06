import numpy as np 

class LinearRegression:
    def __init__(self, l_rate=0.001, n_iters=1000):
        self.l_rate = l_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            self.weights -= self.l_rate * dw
            self.bias -= self.l_rate * db


    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

    
