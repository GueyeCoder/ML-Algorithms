import numpy as np

class Perceptron:

    def __init__(self, l_rate=0.01, n_iters=1000):
        self.l_rate = l_rate
        self.n_iters = n_iters
        self.activation_func = self._uinit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        _, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.array([1 if i >= 0 else 0 for i in y])
        for _ in range(self.n_iters):
            for ind, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_func(linear_output)
                update = self.l_rate * (y_[ind] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation_func(linear_output)
        return y_pred

    def _uinit_step_func(self, x):
        return np.where(x>=0, 1, 0)