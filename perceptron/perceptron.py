import numpy as np

class Perceptron:

    def __init__(self, l_rate=0.01, n_iters=1000):
        self.l_rate = l_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func  
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        
        y_ = np.where(y <= 0, 0, 1)  

        for i in range(self.n_iters):
            errors = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_func(linear_output)

                update = self.l_rate * (y_[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

                if update != 0:
                    errors += 1

            if errors == 0:
                print(f"Converged after {i+1} iterations")
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation_func(linear_output)
        return y_pred

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)
