import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from naive_bayes import NaiveBayes

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

if __name__ == "__main__":
    # Génération des données
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    
    # Entraînement du modèle
    naive_bayes = NaiveBayes()
    naive_bayes.fit(X_train, y_train)
    
    # Prédictions
    predictions = naive_bayes.predict(X_test)
    
    # Affichage des résultats
    print(f"Naive Bayes classification accuracy: {accuracy(y_test, predictions):.4f}")
    print(f"Nombre d'échantillons de test: {len(y_test)}")
    print(f"Nombre de prédictions correctes: {np.sum(y_test == predictions)}")
