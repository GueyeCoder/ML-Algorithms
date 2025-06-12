import numpy as np  

class LogisticRegression:
    def __init__(self, l_rate=0.001, n_iters=1000):
        self.l_rate = l_rate
        self.n_iters = n_iters
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weight) + self.bias
            y_predicted = self._sigmoid(linear_model)
            dw = (1/n_samples) * np.dot(X.T, (y_predicted-y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            self.weight -= self.l_rate * dw
            self.bias -= self.l_rate * db


    def predict(self, X):
        linear_model = np.dot(X, self.weight) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i>0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))