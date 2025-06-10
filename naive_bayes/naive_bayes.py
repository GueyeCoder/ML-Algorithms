import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]  
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)  
    
    def _predict(self, x):
        posteriors = []  
        for index, c in enumerate(self._classes):
            prior = np.log(self._priors[index])
            class_conditional = np.sum(np.log(self._proba_df(index, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]

    def _proba_df(self, class_index, x):
        mean = self._mean[class_index]
        var = self._var[class_index]
        
        # Éviter la division par zéro
        var = np.where(var == 0, 1e-6, var)
        
        # Formule de la densité de probabilité gaussienne
        numerator = np.exp(-(x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)  
        
        return numerator / denominator  