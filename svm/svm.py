import numpy as np

class SVM:
    def __init__(self, l_rate=0.001, lambda_param=0.02, n_iters=1000):
        self.l_rate = l_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for ind, x_i in enumerate(X):
                condition = y_[ind] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.l_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.l_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[ind]))
                    self.b -= self.l_rate * y_[ind]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)